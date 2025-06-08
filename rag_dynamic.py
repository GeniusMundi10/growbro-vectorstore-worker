"""
rag_dynamic.py
Dynamic, multi-tenant RAG agent for serving multiple AIs/clients using rag_utils.py and Supabase.
"""

from supabase_client import supabase
from rag_utils import (
    generate_prompt_template,
    create_chat_prompt_template,
    get_embeddings,
    load_faiss_vectorstore,
    get_hybrid_retriever,
    get_cross_encoder_reranker,
    get_compression_retriever,
    create_conversational_retrieval_chain
)
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import os

class DynamicRAGAgent:
    """
    DynamicRAGAgent is the main interface for multi-tenant RAG.
    Only use get_response() for answering and extract_and_build_vectorstore() for building vectorstore.
    All other methods are internal.
    """
    def __init__(self, ai_id, memory=None, auto_build_vectorstore=False):
        self.ai_id = ai_id
        self.memory = memory or ConversationBufferMemory(
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
            return_messages=True
        )
        self.config = self._fetch_config()
        self.llm = self._init_llm()
        # Versioned vectorstore cache
        if not hasattr(DynamicRAGAgent, '_vectorstore_cache'):
            DynamicRAGAgent._vectorstore_cache = {}
        if not hasattr(DynamicRAGAgent, '_vectorstore_version'):
            DynamicRAGAgent._vectorstore_version = {}
        from rag_utils import download_faiss_index_from_supabase, load_faiss_vectorstore, get_embeddings, get_vectorstore_version_from_supabase
        SUPABASE_BUCKET = "vectorstores"
        from supabase_client import SUPABASE_URL, SUPABASE_KEY
        try:
            current_version = get_vectorstore_version_from_supabase(self.ai_id, SUPABASE_URL, SUPABASE_BUCKET)
        except Exception as e:
            print(f"[DynamicRAGAgent] Could not fetch vectorstore version from Supabase: {e}")
            current_version = None
        cached_version = DynamicRAGAgent._vectorstore_version.get(self.ai_id)
        if self.ai_id not in DynamicRAGAgent._vectorstore_cache or current_version != cached_version:
            try:
                download_faiss_index_from_supabase(self.ai_id, SUPABASE_URL, SUPABASE_BUCKET)
                vectorstore_path = f"faiss_index_{self.ai_id}"
                embeddings = get_embeddings()
                self.vectorstore = load_faiss_vectorstore(vectorstore_path, embeddings)
                if self.vectorstore:
                    DynamicRAGAgent._vectorstore_cache[self.ai_id] = self.vectorstore
                    DynamicRAGAgent._vectorstore_version[self.ai_id] = current_version
            except Exception as e:
                print(f"[DynamicRAGAgent] Could not download or load vectorstore from Supabase: {e}")
                self.vectorstore = None
        else:
            self.vectorstore = DynamicRAGAgent._vectorstore_cache[self.ai_id]
        self.retriever = self._setup_retriever() if self.vectorstore else None
        self.prompt_template = self._build_prompt()
        self.chain = self._build_chain() if self.retriever else None

    def is_ready(self):
        """Return True if agent is ready to answer (vectorstore, retriever, chain all set)."""
        return self.vectorstore is not None and self.retriever is not None and self.chain is not None

    def extract_and_build_vectorstore(self, force_rebuild=False):
        """
        Aggregate and embed all relevant knowledge sources for this AI (websites, ai_links, ai_file), and build the FAISS vectorstore.
        Only builds if not present or if force_rebuild=True. Otherwise, loads existing vectorstore.
        Integrates Supabase Storage for persistent FAISS index if env vars are set.
        """
        import os
        from rag_utils import (
            extract_website_text_with_firecrawl, generic_create_vectorstore, load_faiss_vectorstore,
            get_embeddings, get_text_splitter, download_faiss_index_from_supabase, upload_faiss_index_to_supabase, extract_file_text
        )
        from langchain_core.documents import Document
        from supabase_client import SUPABASE_URL, SUPABASE_KEY, supabase
        import pickle
        vectorstore_path = f"faiss_index_{self.ai_id}"
        faiss_index_file = os.path.join(vectorstore_path, "index.faiss")
        embeddings = get_embeddings()
        SUPABASE_BUCKET = "vectorstores"

        # Download from Supabase if not present locally
        if not os.path.exists(faiss_index_file) and SUPABASE_URL and SUPABASE_KEY:
            try:
                print(f"[DynamicRAGAgent] Attempting to download FAISS index for {self.ai_id} from Supabase Storage bucket '{SUPABASE_BUCKET}'...")
                download_faiss_index_from_supabase(self.ai_id, SUPABASE_URL, SUPABASE_BUCKET)
                print(f"[DynamicRAGAgent] Downloaded FAISS index for {self.ai_id} from Supabase Storage.")
            except Exception as e:
                print(f"[DynamicRAGAgent] Warning: Could not download FAISS index from Supabase: {e}")
        if os.path.exists(faiss_index_file) and not force_rebuild:
            print(f"[DynamicRAGAgent] Vectorstore already exists at {vectorstore_path}. Loading...")
            self.vectorstore = load_faiss_vectorstore(vectorstore_path, embeddings)
            self.retriever = self._setup_retriever()
            self.chain = self._build_chain()
            print("[DynamicRAGAgent] Vectorstore loaded.")
            return

        # --- Aggregate all sources ---
        documents = []

        # 1. Website (mandatory)
        website_urls = []
        if self.config.get("website"):
            if isinstance(self.config["website"], str):
                website_urls = [u.strip() for u in self.config["website"].split(",") if u.strip()]
            elif isinstance(self.config["website"], list):
                website_urls = self.config["website"]
        if website_urls:
            print(f"[DynamicRAGAgent] Extracting website text for URLs: {website_urls}")
            documents += extract_website_text_with_firecrawl(website_urls)
        else:
            print("[DynamicRAGAgent] No website URLs found in config (this should not happen).")

        # 2. ai_links (optional)
        try:
            ai_links_res = supabase.table("ai_website").select("*").eq("user_id", self.config["user_id"]).execute()
            link_urls = [row["url"] for row in (ai_links_res.data or []) if row.get("url")]
            if link_urls:
                print(f"[DynamicRAGAgent] Extracting text for ai_links URLs: {link_urls}")
                documents += extract_website_text_with_firecrawl(link_urls)
            else:
                print("[DynamicRAGAgent] No ai_links URLs found for this user.")
        except Exception as e:
            print(f"[DynamicRAGAgent] Error fetching ai_links: {e}")

        # 3. ai_file (optional, use 'url' to download and extract text)
        import requests
        try:
            ai_files_res = supabase.table("ai_file").select("*").eq("user_id", self.config["user_id"]).execute()
            file_rows = ai_files_res.data or []
            for file_row in file_rows:
                file_url = file_row.get("url")
                if file_url:
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
            if not file_rows:
                print("[DynamicRAGAgent] No ai_file entries found for this user.")
        except Exception as e:
            print(f"[DynamicRAGAgent] Error fetching ai_file: {e}")

        # --- Chunk, embed, and save vectorstore ---
        if not documents:
            raise RuntimeError("No documents found for embedding from any source.")
        text_splitter = get_text_splitter()
        splits = text_splitter.split_documents(documents)
        os.makedirs(vectorstore_path, exist_ok=True)
        with open(os.path.join(vectorstore_path, "splits.pkl"), "wb") as f:
            pickle.dump(splits, f)
        print(f"[DynamicRAGAgent] Saved splits for BM25 at {vectorstore_path}/splits.pkl")
        print(f"[DynamicRAGAgent] Creating vectorstore at {vectorstore_path}")
        self.vectorstore = generic_create_vectorstore(splits, embeddings, vectorstore_path)
        self.retriever = self._setup_retriever()
        self.chain = self._build_chain()
        print("[DynamicRAGAgent] Vectorstore and retriever rebuilt.")
        # --- Upload new/updated FAISS index to Supabase Storage ---
        if SUPABASE_URL and SUPABASE_KEY:
            try:
                print(f"[DynamicRAGAgent] Uploading FAISS index for {self.ai_id} to Supabase Storage bucket '{SUPABASE_BUCKET}'...")
                upload_faiss_index_to_supabase(self.ai_id, SUPABASE_URL, SUPABASE_BUCKET, SUPABASE_KEY, local_dir=".")
                print(f"[DynamicRAGAgent] Uploaded FAISS index for {self.ai_id} to Supabase Storage.")
                # Write and upload version.txt
                import datetime, os
                vectorstore_path = f"faiss_index_{self.ai_id}"
                version_txt_path = os.path.join(vectorstore_path, "version.txt")
                version = datetime.datetime.utcnow().isoformat()
                with open(version_txt_path, "w") as f:
                    f.write(version)
                upload_faiss_index_to_supabase(self.ai_id, SUPABASE_URL, SUPABASE_BUCKET, SUPABASE_KEY, local_dir=".")
                print(f"[DynamicRAGAgent] Uploaded version.txt for {self.ai_id} to Supabase Storage.")
            except Exception as e:
                print(f"[DynamicRAGAgent] Warning: Could not upload FAISS index to Supabase: {e}")

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
        vectorstore_path = f"faiss_index_{self.ai_id}"
        embeddings = get_embeddings()
        try:
            return load_faiss_vectorstore(vectorstore_path, embeddings)
        except Exception as e:
            print(f"[DynamicRAGAgent] Could not load vectorstore for {self.ai_id}: {e}")
            return None

    def _setup_retriever(self):
        if not self.vectorstore:
            raise RuntimeError("No vectorstore loaded.")
        import os, pickle
        vectorstore_path = f"faiss_index_{self.ai_id}"
        splits_path = os.path.join(vectorstore_path, "splits.pkl")
        if not os.path.exists(splits_path):
            raise RuntimeError(f"splits.pkl not found at {splits_path}. Please rebuild vectorstore.")
        with open(splits_path, "rb") as f:
            splits = pickle.load(f)
        # Hybrid retriever (BM25 + dense)
        base_retriever = get_hybrid_retriever(self.vectorstore, splits)
        reranker = get_cross_encoder_reranker()
        return get_compression_retriever(base_retriever, reranker)

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
