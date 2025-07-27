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
        # Initialize Pinecone serverless vectorstore (no local caching needed)
        from pinecone_serverless_utils import check_index_exists, PineconeServerlessRetriever
        
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
        """Return True if agent is ready to answer (vectorstore, retriever, chain all set)."""
        return self.vectorstore is not None and self.retriever is not None and self.chain is not None

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
            PineconeServerlessRetriever,
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

    def _init_llm(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("No OpenAI API key found for this AI.")
        return ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-4.1-nano",
            streaming=True,
            temperature=0.7
        )

    def _load_vectorstore(self):
        """Load Pinecone vectorstore for this AI."""
        try:
            from pinecone_serverless_utils import get_pinecone_retriever
            return get_pinecone_retriever(self.ai_id)
        except Exception as e:
            print(f"[DynamicRAGAgent] Could not load Pinecone vectorstore for {self.ai_id}: {e}")
            return None

    def _setup_retriever(self):
        if not self.vectorstore:
            raise RuntimeError("No vectorstore loaded.")
        # Pinecone serverless retriever is already initialized
        # Return the vectorstore as it implements the retriever interface
        return self.vectorstore

    def _build_prompt(self):
        """
        Build a dynamic prompt using business_info, ai_greeting, ai_services, ai_resource_link_file (all null-safe).
        Includes greetings, services, resource links, and core company info.
        """
        from supabase_client import supabase
        from rag_utils import create_chat_prompt_template
        company_info = {
            "bot_name": self.config.get("agent_type", "AI Assistant"),
            "company_name": self.config.get("company_name", "Company"),
            "website_url": self.config.get("website", ""),
            "contact_url": self.config.get("email", ""),
            "language": self.config.get("language", "English"),
            "extra_instructions": self.config.get("extra_instructions", "")
        }

        # 1. Greetings (optional)
        try:
            greetings_res = supabase.table("ai_greeting").select("*").eq("user_id", self.config["user_id"]).execute()
            greetings = [row["message"] for row in (greetings_res.data or []) if row.get("message")]
            company_info["greetings"] = "\n".join(greetings) if greetings else ""
        except Exception as e:
            print(f"[DynamicRAGAgent] Error fetching ai_greeting: {e}")
            company_info["greetings"] = ""

        # 2. Services (optional)
        try:
            services_res = supabase.table("ai_services").select("*").eq("user_id", self.config["user_id"]).execute()
            if services_res.data:
                service = services_res.data[0]
                company_info["business_services"] = service.get("business_services", "")
                company_info["differentiation"] = service.get("differentiation", "")
                company_info["profitable_line_items"] = service.get("profitable_line_items", "")
                company_info["best_sales_lines"] = service.get("best_sales_lines", "")
            else:
                company_info["business_services"] = ""
                company_info["differentiation"] = ""
                company_info["profitable_line_items"] = ""
                company_info["best_sales_lines"] = ""
        except Exception as e:
            print(f"[DynamicRAGAgent] Error fetching ai_services: {e}")
            company_info["business_services"] = ""
            company_info["differentiation"] = ""
            company_info["profitable_line_items"] = ""
            company_info["best_sales_lines"] = ""

        # 3. Resource Links (optional, grouped by category from CSV)
        import csv
        import requests
        from collections import defaultdict
        try:
            resource_links_res = supabase.table("ai_resource_link_file").select("*").eq("user_id", self.config["user_id"]).execute()
            csv_urls = [row["url"] for row in (resource_links_res.data or []) if row.get("url")]
            category_resources = defaultdict(list)
            for csv_url in csv_urls:
                try:
                    resp = requests.get(csv_url)
                    resp.raise_for_status()
                    decoded = resp.content.decode("utf-8")
                    reader = csv.DictReader(decoded.splitlines())
                    for row in reader:
                        title = row.get("Title")
                        url = row.get("URL")
                        category = row.get("Category")
                        if title and url and category:
                            for cat in [c.strip() for c in category.split(",")]:
                                category_resources[cat].append(f"- [{title}]({url})")
                except Exception as e:
                    print(f"[DynamicRAGAgent] Error downloading/parsing resource CSV: {e}")
            # Format grouped resources for the prompt
            if category_resources:
                grouped = []
                for cat, links in category_resources.items():
                    grouped.append(f"{cat}:\n" + "\n".join(links))
                company_info["resource_links"] = "\n\n".join(grouped)
            else:
                company_info["resource_links"] = ""
        except Exception as e:
            print(f"[DynamicRAGAgent] Error fetching ai_resource_link_file: {e}")
            company_info["resource_links"] = ""

        # You may want to extend your base template to include these fields for richer bot persona
        return create_chat_prompt_template(company_info)

    def _build_chain(self):
        return create_conversational_retrieval_chain(
            llm=self.llm,
            retriever=self.retriever,
            prompt_template=self.prompt_template,
            memory=self.memory
        )

    def get_response(self, question, chat_history=None):
        """
        Get an answer for a user question. Only call this if is_ready() is True.
        Raises a clear error if not ready (e.g. vectorstore missing).
        """
        if not self.is_ready():
            raise RuntimeError("DynamicRAGAgent is not ready. Please build the vectorstore first using extract_and_build_vectorstore().")
        if chat_history:
            self.memory.chat_memory.messages = chat_history
        # Retrieve context
        docs = self.retriever.invoke(question)
        context = "\n".join([doc.page_content for doc in docs])
        chain_inputs = {
            "question": question,
            "context": context,
            "chat_history": self.memory.buffer if hasattr(self.memory, "buffer") else ""
        }
        response = self.chain.invoke(chain_inputs)
        answer = response["answer"] if isinstance(response, dict) and "answer" in response else str(response)
        self.memory.save_context({"question": question}, {"answer": answer})
        return answer

# Example usage:
# agent = DynamicRAGAgent(ai_id="your_ai_id_here")
# response = agent.get_response("What are your business hours?")
