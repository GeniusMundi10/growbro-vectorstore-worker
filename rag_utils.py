"""
rag_utils.py
Reusable, generic utilities for RAG pipelines (data extraction, vectorstore creation, prompt templating).
"""


import requests
from langchain_core.documents import Document
import mimetypes

# --- File extraction utility ---
def extract_file_text(file_path):
    """
    Extract text from a file (PDF, TXT, DOCX, etc.).
    Returns a string or None if extraction fails.
    """
    import os
    if not os.path.exists(file_path):
        print(f"[extract_file_text] File not found: {file_path}")
        return None
    mime, _ = mimetypes.guess_type(file_path)
    try:
        if mime and mime.startswith("text"):
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif mime == "application/pdf" or file_path.lower().endswith(".pdf"):
            try:
                from PyPDF2 import PdfReader
            except ImportError:
                print("[extract_file_text] PyPDF2 not installed. Skipping PDF extraction.")
                return None
            try:
                reader = PdfReader(file_path)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as e:
                print(f"[extract_file_text] PDF extraction failed: {e}")
                return None
        elif file_path.lower().endswith(".docx"):
            try:
                import docx
            except ImportError:
                print("[extract_file_text] python-docx not installed. Skipping DOCX extraction.")
                return None
            try:
                doc = docx.Document(file_path)
                return "\n".join([p.text for p in doc.paragraphs])
            except Exception as e:
                print(f"[extract_file_text] DOCX extraction failed: {e}")
                return None
        else:
            print(f"[extract_file_text] Unsupported file type: {file_path}")
            return None
    except Exception as e:
        print(f"[extract_file_text] Extraction error: {e}")
        return None

# Firecrawl SDK
try:
    from firecrawl import FirecrawlApp, ScrapeOptions
except ImportError:
    FirecrawlApp = None
    ScrapeOptions = None
import os
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate

def aggregate_crawl_analytics(website_analytics, link_analytics, file_analytics):
    """
    Aggregate analytics from all sources.
    - total_pages_crawled: count of unique URLs from website and ai_links only
    - files_indexed: count of ai_files (file_analytics['files_indexed'] if present, else 0)
    Returns a dict: {'total_pages_crawled': int, 'urls_crawled': list, 'files_indexed': int}
    """
    all_urls = set()
    for analytics in (website_analytics, link_analytics):
        if analytics and 'urls_crawled' in analytics:
            all_urls.update(analytics['urls_crawled'])
    files_indexed = 0
    if file_analytics and 'files_indexed' in file_analytics:
        files_indexed = file_analytics['files_indexed']
    return {
        'total_pages_crawled': len(all_urls),
        'urls_crawled': list(all_urls),
        'files_indexed': files_indexed
    }


# --- FAISS index upload/download utilities for Supabase Storage ---
import requests

def get_vectorstore_version_from_supabase(ai_id, supabase_url, bucket):
    """Fetches version.txt from Supabase Storage for the given ai_id and returns its contents as the version string."""
    version_url = f"{supabase_url}/storage/v1/object/public/{bucket}/faiss_index_{ai_id}/version.txt"
    resp = requests.get(version_url)
    if resp.status_code == 200:
        return resp.text.strip()
    else:
        raise RuntimeError(f"Failed to fetch vectorstore version for {ai_id}: {resp.status_code}")

def download_faiss_index_from_supabase(ai_id, supabase_url, bucket, local_dir="."):
    """
    Downloads all FAISS index files for a given ai_id from Supabase Storage to a local directory (default: current directory).
    Handles: index.faiss, index.pkl, splits.pkl
    Returns the local directory path containing the index files.
    Usage:
        local_index_dir = download_faiss_index_from_supabase(
            ai_id,
            supabase_url="https://your-project.supabase.co",
            bucket="your-bucket"
        )
    """
    import requests, os
    files = ["index.faiss", "index.pkl", "splits.pkl"]
    local_path = os.path.join(local_dir, f"faiss_index_{ai_id}")
    print(f"[FAISS Download] Downloading to {local_path}")
    os.makedirs(local_path, exist_ok=True)
    for fname in files:
        storage_url = f"{supabase_url}/storage/v1/object/public/{bucket}/faiss_index_{ai_id}/{fname}"
        out_file = os.path.join(local_path, fname)
        resp = requests.get(storage_url)
        if resp.status_code == 200:
            with open(out_file, "wb") as f:
                f.write(resp.content)
        else:
            raise RuntimeError(f"Failed to download {fname} for ai_id {ai_id}: {resp.status_code}")
    return local_path

def upload_faiss_index_to_supabase(ai_id, supabase_url, bucket, supabase_key, local_dir="."):
    """
    Uploads all FAISS index files for a given ai_id from a local directory to Supabase Storage (default: current directory).
    Handles: index.faiss, index.pkl, splits.pkl
    Requires the Supabase service role key for authentication (pass SUPABASE_KEY from supabase_client.py).
    Usage:
        upload_faiss_index_to_supabase(
            ai_id,
            supabase_url="https://your-project.supabase.co",
            bucket="vectorstores",
            supabase_key=SUPABASE_KEY
        )
    """
    import requests, os
    files = ["index.faiss", "index.pkl", "splits.pkl", "version.txt"]
    local_path = os.path.join(local_dir, f"faiss_index_{ai_id}")
    print(f"[FAISS Upload] Uploading from {local_path}")
    headers = {
        "Authorization": f"Bearer {supabase_key}",
        "Content-Type": "application/octet-stream"
    }
    for fname in files:
        file_path = os.path.join(local_path, fname)
        if not os.path.exists(file_path):
            if fname == "version.txt":
                continue  # Only upload version.txt if it exists
            raise RuntimeError(f"Local file missing: {file_path}")
        storage_url = f"{supabase_url}/storage/v1/object/{bucket}/faiss_index_{ai_id}/{fname}"
        with open(file_path, "rb") as f:
            resp = requests.put(storage_url, headers=headers, data=f)
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Failed to upload {fname} for ai_id {ai_id}: {resp.status_code} - {resp.text}")
    return True




def extract_website_text_with_firecrawl(urls, min_words=10, firecrawl_api_key=None, formats=['markdown'], limit=20, return_analytics=False, session_cookie=None, deep_crawl=False):
    """
    Extracts website text using Firecrawl. Falls back to generic_extract_website_text if Firecrawl fails or is unavailable.
    Returns: list of LangChain Document objects
    If return_analytics=True, returns (documents, analytics_dict) where analytics_dict has keys 'pages_crawled', 'urls_crawled'.
    """
    api_key = firecrawl_api_key or os.environ.get("FIRECRAWL_API_KEY")
    all_documents = []
    urls_crawled = []
    if deep_crawl and len(urls) > 0:
        url = urls[0]
        try:
            # Ensure URL has scheme
            if not url.startswith("http://") and not url.startswith("https://"):
                url = "https://" + url
            if session_cookie:
                # Use Firecrawl REST API with session cookie
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Cookie": session_cookie
                }
                data = {
                    "url": url,
                    "formats": formats,
                    "limit": limit
                    # No maxDepth parameter for deep crawl
                }
                resp = requests.post("https://api.firecrawl.dev/v1/scrape", headers=headers, json=data, timeout=60)
                resp.raise_for_status()
                result = resp.json()
                for item in result.get('data', []):
                    content = item.get('markdown') or item.get('html') or ""
                    metadata = item.get('metadata', {})
                    all_documents.append(Document(page_content=content, metadata=metadata))
                    page_url = metadata.get('url') or url
                    urls_crawled.append(page_url)
            else:
                # Use Firecrawl SDK for public crawling
                if FirecrawlApp is None or ScrapeOptions is None:
                    raise ImportError("Firecrawl SDK is not installed.")
                app = FirecrawlApp(api_key=api_key)
                crawl_result = app.crawl_url(url, limit=limit, scrape_options=ScrapeOptions(formats=formats))  # No maxDepth for deep crawl
                if hasattr(crawl_result, 'status') and crawl_result.status == 'completed':
                    status = crawl_result
                else:
                    crawl_id = crawl_result.id
                    while True:
                        status = app.check_crawl_status(crawl_id)
                        if status.status == 'completed':
                            break
                        elif status.status == 'failed':
                            raise RuntimeError(f"Firecrawl crawl failed: {status}")
                        time.sleep(3)
                for item in status.data:
                    content = getattr(item, 'markdown', None) or getattr(item, 'html', None) or ""
                    metadata = getattr(item, 'metadata', {})
                    all_documents.append(Document(page_content=content, metadata=metadata))
                    page_url = metadata.get('url') or url
                    urls_crawled.append(page_url)
        except Exception as e:
            print(f"[Firecrawl] Error deep crawling {url}: {e}. Please Contact Growbro")
    elif not deep_crawl:
        for url in urls:
            try:
                # Ensure URL has scheme
                if not url.startswith("http://") and not url.startswith("https://"):
                    url = "https://" + url
                if session_cookie:
                    # Use Firecrawl REST API with session cookie
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "Cookie": session_cookie
                    }
                    data = {
                        "url": url,
                        "formats": formats,
                        "limit": limit,
                        "maxDepth": 0  # Only crawl this page, do not follow links
                    }
                    resp = requests.post("https://api.firecrawl.dev/v1/scrape", headers=headers, json=data, timeout=60)
                    resp.raise_for_status()
                    result = resp.json()
                    for item in result.get('data', []):
                        content = item.get('markdown') or item.get('html') or ""
                        metadata = item.get('metadata', {})
                        all_documents.append(Document(page_content=content, metadata=metadata))
                        page_url = metadata.get('url') or url
                        urls_crawled.append(page_url)
                else:
                    # Use Firecrawl SDK for public crawling
                    if FirecrawlApp is None or ScrapeOptions is None:
                        raise ImportError("Firecrawl SDK is not installed.")
                    app = FirecrawlApp(api_key=api_key)
                    crawl_result = app.crawl_url(url, limit=limit, scrape_options=ScrapeOptions(formats=formats), maxDepth=0)  # Only crawl this page
                    if hasattr(crawl_result, 'status') and crawl_result.status == 'completed':
                        status = crawl_result
                    else:
                        crawl_id = crawl_result.id
                        while True:
                            status = app.check_crawl_status(crawl_id)
                            if status.status == 'completed':
                                break
                            elif status.status == 'failed':
                                raise RuntimeError(f"Firecrawl crawl failed: {status}")
                            time.sleep(3)
                    for item in status.data:
                        content = getattr(item, 'markdown', None) or getattr(item, 'html', None) or ""
                        metadata = getattr(item, 'metadata', {})
                        all_documents.append(Document(page_content=content, metadata=metadata))
                        page_url = metadata.get('url') or url
                        urls_crawled.append(page_url)
            except Exception as e:
                print(f"[Firecrawl] Error crawling {url}: {e}. Please Contact Growbro")
                # Fallback for this URL only
                # Optionally could call generic_extract_website_text here and add to docs/analytics
    if return_analytics:
        # Remove duplicates and nulls
        urls_crawled = [u for u in set(urls_crawled) if u]
        return all_documents, {"pages_crawled": len(urls_crawled), "urls_crawled": urls_crawled}
    return all_documents




def generic_create_vectorstore(documents, embeddings, vectorstore_path, chunk_size=3000, chunk_overlap=300):
    """
    Create and save a FAISS vectorstore from documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(vectorstore_path)
    return vectorstore


def get_generic_base_prompt():
    """
    Returns a sales-focused, conversion-driven prompt template for a business AI assistant.
    All company/bot fields are dynamic and must be filled via .format(**company_info).
    Placeholders: {bot_name}, {company_name}, {website_url}, {contact_url}, {language}, {extra_instructions},
    {greetings}, {business_services}, {differentiation}, {profitable_line_items}, {best_sales_lines}, {resource_links}
    """
    return (
        "If any section below is empty, ignore it and do not mention it to the user.\n"
        "You are {bot_name}, the primary sales lead generator and customer representative for {company_name}.\n"
        "Your main mission is to qualify, nurture, and convert every website visitor into a potential customer or sales lead.\n"
        "Always be proactive, persuasive, and positive in your responses.\n"
        "\n"
        "Company website: {website_url}\n"
        "Contact: {contact_url}\n"
        "Language: {language}\n"
        "{extra_instructions}\n"
        "\n"
        "### Initial Greeting\n"
        "When a user first interacts with you, choose one of the following greetings (if available) to start the conversation. If none are present, skip this step:\n"
        "{greetings}\n"
        "\n"
        "### Business Services\n"
        "If provided, our company offers the following services:\n"
        "- {business_services}\n"
        "\n"
        "### Unique Selling Points\n"
        "If provided, emphasize these unique selling points to differentiate us from competitors:\n"
        "- {differentiation}\n"
        "\n"
        "### Most Valuable Offerings\n"
        "If provided, promote these profitable line items to users:\n"
        "- {profitable_line_items}\n"
        "\n"
        "### Persuasive Sales Lines\n"
        "If provided, use these phrases to persuade and build trust with users:\n"
        "- {best_sales_lines}\n"
        "\n"
        "### Relevant Resource Links\n"
        "If provided, recommend these resources to users when relevant to their query or category:\n"
        "- {resource_links}\n"
        "\n"
        "---\n"
        "Instructions for every conversation:\n"
        "- If greetings are present, choose one at random or rotate between them for each new conversation.\n"
        "- If any section above is empty, simply ignore it and do not reference it in your response.\n"
        "- Quickly identify the user's needs, pain points, or goals.\n"
        "- Clearly explain how {company_name} and its services can address those needs.\n"
        "- Use the unique differentiators and sales lines to persuade and build trust.\n"
        "- If a free trial, demo, or profitable offering is available, highlight it and encourage the user to try it.\n"
        "- Always encourage the user to take the next step: sign up, book a demo, or contact the team.\n"
        "- If resource links are provided, recommend them when relevant to the user's query.\n"
        "- Whenever possible, capture contact information or guide the user to the contact: {contact_url}.\n"
        "- NEVER invent or guess facts, features, URLs, or company details. ONLY use the information provided above.\n"
        "- Be concise, warm, and engaging. Always respond in {language}.\n"
        "- If you don't know the answer, politely direct the user to contact the team for further assistance.\n"
        "{extra_instructions}"
    )

def generate_prompt_template(company_info, base_template=None):
    """
    Generate a dynamic prompt for any company/bot.
    - company_info: dict with keys matching placeholders in the template (see get_generic_base_prompt)
    - base_template: (optional) override the default generic template
    Returns: formatted prompt string
    """
    if base_template is None:
        base_template = get_generic_base_prompt()
    return base_template.format(**company_info)


def create_chat_prompt_template(company_info, base_template=None, human_template=None):
    """
    Returns a ChatPromptTemplate with dynamic system prompt and standard human prompt.
    - company_info: dict for the system prompt
    - base_template: (optional) custom system prompt template string
    - human_template: (optional) custom human message template; default includes context, question, chat_history
    """
    if base_template is None:
        base_template = get_generic_base_prompt()
    system_prompt = base_template.format(**company_info)
    if human_template is None:
        human_template = "Context: {context}\nQuestion: {question}\nChat History: {chat_history}"
    return ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", human_template)
    ])


def get_embeddings(model_name="BAAI/bge-large-en-v1.5"):
    """
    Returns a HuggingFaceEmbeddings object for the specified model.
    """
    
    return HuggingFaceEmbeddings(model_name=model_name)


def get_text_splitter(chunk_size=1000, chunk_overlap=100):
    """
    Returns a RecursiveCharacterTextSplitter with the specified parameters.
    """
    
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def create_faiss_vectorstore(texts, embeddings, path):
    """
    Create and save a FAISS vectorstore from texts and embeddings.
    """
    
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(path)
    return vectorstore


def load_faiss_vectorstore(path, embeddings):
    """
    Load a FAISS vectorstore from disk.
    """
    
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def get_hybrid_retriever(vectorstore, docs, bm25_k=3, dense_k=3):
    """
    Returns an EnsembleRetriever combining BM25 and dense retriever.
    """
    
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = bm25_k
    dense = vectorstore.as_retriever(search_kwargs={"k": dense_k})
    return EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])


def get_cross_encoder_reranker(model_name="BAAI/bge-reranker-base"):
    """
    Returns a CrossEncoderReranker using the specified model.
    """
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    cross_encoder = HuggingFaceCrossEncoder(model_name=model_name)
    return CrossEncoderReranker(model=cross_encoder, top_n=6)


def get_compression_retriever(base_retriever, reranker):
    """
    Wraps a retriever with a reranker for contextual compression.
    """
    from langchain.retrievers import ContextualCompressionRetriever
    return ContextualCompressionRetriever(base_compressor=reranker, base_retriever=base_retriever)


def create_conversational_retrieval_chain(llm, retriever, prompt_template, memory=None, **kwargs):
    """
    Returns a ConversationalRetrievalChain with all components dynamic.
    """
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        condense_question_prompt=ChatPromptTemplate.from_messages([
            ("system", "Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language."),
            ("human", "Chat History: {chat_history}\nFollow Up Input: {question}\nStandalone question:"),
        ]),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt_template},
        verbose=True,
        **kwargs
    )
