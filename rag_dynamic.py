"""
rag_dynamic.py
Dynamic, multi-tenant RAG agent for serving multiple AIs/clients using rag_utils.py and Supabase.
"""

from supabase_client import supabase
from rag_utils import (
    generate_prompt_template,
    create_chat_prompt_template,
    get_text_splitter,
    extract_website_text_with_firecrawl,
    extract_file_text,
    create_conversational_retrieval_chain,
    aggregate_crawl_analytics
)
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import os

def save_splits_to_supabase(ai_id: str, splits):
    """
    Save document splits to Supabase storage as pickle file for BM25 hybrid retrieval.
    """
    try:
        import pickle
        import io
        
        print(f"[Supabase] Saving {len(splits)} document splits for AI {ai_id}")
        
        # Serialize splits to bytes
        splits_data = pickle.dumps(splits)
        
        # Create file-like object from bytes
        file_obj = io.BytesIO(splits_data)
        
        # Upload to Supabase storage
        file_path = f"vectorstores/{ai_id}/splits.pkl"
        
        # Delete existing file if it exists
        try:
            supabase.storage.from_("vectorstores").remove([file_path])
        except:
            pass  # File doesn't exist, that's ok
        
        # Upload new file
        response = supabase.storage.from_("vectorstores").upload(
            file_path, 
            file_obj.getvalue(),  # Pass bytes, not BytesIO
            file_options={"content-type": "application/octet-stream"}
        )
        
        print(f"[Supabase] Successfully saved splits.pkl for AI {ai_id}")
        return True
        
    except Exception as e:
        print(f"[Supabase] Error saving splits for AI {ai_id}: {e}")
        return False

class DynamicRAGAgent:
    """
    DynamicRAGAgent is the main interface for multi-tenant RAG.
    Only use get_response() for answering and extract_and_build_vectorstore() for building vectorstore.
    All other methods are internal.
    """
    def __init__(self, ai_id, memory=None, auto_build_vectorstore=False, session_cookie=None):
        self.session_cookie = session_cookie
        self.ai_id = ai_id
        self.memory = memory or ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True
        )
        self.config = self._fetch_config()
        self.llm = self._init_llm()
        
        # Import check_index_exists from pinecone_serverless_utils
        from pinecone_serverless_utils import check_index_exists
        
        # Check if Pinecone index exists for this AI
        if check_index_exists(self.ai_id):
            print(f"[DynamicRAGAgent] Pinecone serverless index found for AI {self.ai_id}")
            self.vectorstore = PineconeServerlessRetriever(ai_id=self.ai_id)
        else:
            print(f"[DynamicRAGAgent] No Pinecone index found for AI {self.ai_id}. Will create on first build.")
            self.vectorstore = None
        self.retriever = self._setup_retriever() if self.vectorstore else None
        self.prompt_template = self._build_prompt()
        self.chain = self._build_chain() if self.retriever else None

    def is_ready(self):
        """Return True if agent has been initialized successfully. In growbro-worker, we only
        use this to verify the agent was created, not for RAG functionality."""
        return True

    def extract_and_build_vectorstore(self, force_rebuild=False):
        """
        Aggregate all relevant knowledge sources for this AI (websites, ai_links, ai_file) and create Pinecone serverless vectorstore.
        Uses Pinecone's managed embedding service - no local computation required!
        Only builds if not present or if force_rebuild=True.
        """
        import os
        from langchain_core.documents import Document
        from supabase_client import supabase
        from pinecone_serverless_utils import (
            check_index_exists, 
            upsert_documents_with_lightweight_embeddings, 
            
            delete_vectors_by_ai_id
        )
        
        # Check if Pinecone index already exists and we don't need to rebuild
        if check_index_exists(self.ai_id) and not force_rebuild:
            print(f"[DynamicRAGAgent] Pinecone index already exists for AI {self.ai_id}. Loading...")
            self.vectorstore = PineconeServerlessRetriever(ai_id=self.ai_id)
            self.retriever = self._setup_retriever()
            self.chain = self._build_chain()
            print("[DynamicRAGAgent] Pinecone vectorstore loaded.")
            return
        
        # If force_rebuild, clear existing vectors
        if force_rebuild and check_index_exists(self.ai_id):
            print(f"[DynamicRAGAgent] Force rebuild requested. Clearing existing vectors for AI {self.ai_id}...")
            delete_vectors_by_ai_id(self.ai_id)

        # --- Aggregate all sources and track analytics ---
        documents = []
        website_analytics = None
        link_analytics = None
        file_analytics = None
        file_urls = []

        # 1. Website/URLs logic
        # Always perform deep crawl from website(s), ignore urls_crawled for crawl logic
        website_urls = []
        if self.config.get("website"):
            if isinstance(self.config["website"], str):
                website_urls = [u.strip() for u in self.config["website"].split(",") if u.strip()]
            elif isinstance(self.config["website"], list):
                website_urls = self.config["website"]
        if website_urls:
            print(f"[DynamicRAGAgent] Deep crawling website(s): {website_urls}")
            docs, website_analytics = extract_website_text_with_firecrawl(website_urls, return_analytics=True, session_cookie=self.session_cookie, deep_crawl=True)
            documents += docs
        else:
            print("[DynamicRAGAgent] No website URLs found in config (this should not happen).")

        # 2. ai_links (optional)
        try:
            ai_links_res = supabase.table("ai_website").select("*").eq("user_id", self.config["user_id"]).eq("ai_id", self.ai_id).execute()
            link_urls = [row["url"] for row in (ai_links_res.data or []) if row.get("url")]
            if link_urls:
                print(f"[DynamicRAGAgent] Extracting text for ai_links URLs: {link_urls}")
                # Always shallow crawl for ai_links (single page per link)
                docs, link_analytics = extract_website_text_with_firecrawl(link_urls, return_analytics=True, session_cookie=self.session_cookie, deep_crawl=False)
                documents += docs
            else:
                print("[DynamicRAGAgent] No ai_links URLs found for this user.")
        except Exception as e:
            print(f"[DynamicRAGAgent] Error fetching ai_links: {e}")

        # 3. ai_file (optional, use 'url' to download and extract text)
        import requests
        try:
            ai_files_res = supabase.table("ai_file").select("*").eq("user_id", self.config["user_id"]).eq("ai_id", self.ai_id).execute()
            file_rows = ai_files_res.data or []
            for file_row in file_rows:
                file_url = file_row.get("url")
                if file_url:
                    file_urls.append(file_url)
                    try:
                        resp = requests.get(file_url)
                        resp.raise_for_status()
                        # Save to a temp file for extraction
                        import tempfile
                        import os
                        suffix = os.path.splitext(file_url)[-1]
                        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp_file:
                            tmp_file.write(resp.content)
                            tmp_file.flush()
                            text = extract_file_text(tmp_file.name)
                            if text:
                                documents.append(Document(page_content=text, metadata={"source": file_url}))
                            else:
                                print(f"[DynamicRAGAgent] No text extracted from file: {file_url}")
                    except Exception as e:
                        print(f"[DynamicRAGAgent] Error downloading or extracting file: {file_url}, error: {e}")
                else:
                    print("[DynamicRAGAgent] ai_file row missing url.")
            if file_urls:
                file_analytics = {"files_indexed": len(file_urls)}
            if not file_rows:
                print("[DynamicRAGAgent] No ai_file entries found for this user.")
        except Exception as e:
            print(f"[DynamicRAGAgent] Error fetching ai_file: {e}")

        # --- Aggregate analytics and update Supabase ---
        analytics = aggregate_crawl_analytics(website_analytics, link_analytics, file_analytics)
        # Save analytics after vectorstore upload and after setting vectorstore_ready

        # --- Process and upload documents to Pinecone serverless ---
        if not documents:
            raise RuntimeError("No documents found for embedding from any source.")
        
        text_splitter = get_text_splitter()
        splits = text_splitter.split_documents(documents)
        
        print(f"[DynamicRAGAgent] Creating Pinecone serverless vectorstore with {len(splits)} document chunks")
        print(f"[DynamicRAGAgent] ⚡ Using Pinecone managed embeddings - no local computation!")
        
        # Upload to Pinecone with lightweight embeddings (much faster than heavy FAISS index building)
        upsert_documents_with_lightweight_embeddings(self.ai_id, splits)
        
        # Save splits to Supabase for BM25 hybrid retrieval
        print(f"[DynamicRAGAgent] Saving document splits to Supabase for hybrid retrieval")
        save_splits_to_supabase(self.ai_id, splits)
        
        print("[DynamicRAGAgent] ✅ Pinecone serverless vectorstore created successfully - MUCH faster!")
        try:
            supabase.table("business_info").update({"vectorstore_ready": True}).eq("id", self.ai_id).execute()
            print(f"[DynamicRAGAgent] Set vectorstore_ready=True for {self.ai_id}")
        except Exception as e:
            print(f"[DynamicRAGAgent] Failed to set vectorstore_ready: {e}")
        # Now update analytics
        analytics = aggregate_crawl_analytics(website_analytics, link_analytics, file_analytics)
        try:
            supabase.table("business_info").update({
                "total_pages_crawled": analytics['total_pages_crawled'],
                "urls_crawled": analytics['urls_crawled'],
                "files_indexed": analytics['files_indexed']
            }).eq("id", self.ai_id).execute()
            print(f"[DynamicRAGAgent] Saved crawl analytics for {self.ai_id}: {analytics}")
        except Exception as e:
            print(f"[DynamicRAGAgent] Failed to save crawl analytics: {e}")

    def _fetch_config(self):
        res = supabase.table("business_info").select("*").eq("id", self.ai_id).execute()
        if not res.data or len(res.data) == 0:
            raise ValueError(f"No config found for ai_id={self.ai_id}")
        return res.data[0]

  

  

   

    
