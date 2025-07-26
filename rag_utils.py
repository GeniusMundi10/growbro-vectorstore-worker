"""
rag_utils.py
Essential utility functions for growbro-worker vectorstore creation.
"""

import os
import time
import requests
import mimetypes
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Firecrawl SDK
try:
    from firecrawl import FirecrawlApp, ScrapeOptions
except ImportError:
    FirecrawlApp = None
    ScrapeOptions = None

def extract_file_text(file_path):
    """
    Extract text from a file (PDF, TXT, DOCX, etc.).
    Returns a string or None if extraction fails.
    """
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

def extract_website_text_with_firecrawl(urls, min_words=10, firecrawl_api_key=None, formats=['markdown'], limit=30, return_analytics=False, session_cookie=None, deep_crawl=False):
    """
    Extracts website text using Firecrawl. Falls back to generic extraction if Firecrawl fails.
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
            print(f"[Firecrawl Debug] Deep crawl target url: {url}")
            
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
                    "crawlEntireDomain": True
                }
                
                print(f"[Firecrawl Debug] Starting deep crawl for: {url}")
                resp = requests.post("https://api.firecrawl.dev/v1/crawl", headers=headers, json=data)
                
                if resp.status_code == 200:
                    crawl_job = resp.json()
                    job_id = crawl_job.get("id")
                    if job_id:
                        print(f"[Firecrawl Debug] Crawl job started: {job_id}")
                        # Poll for completion
                        while True:
                            status_resp = requests.get(f"https://api.firecrawl.dev/v1/crawl/{job_id}", headers=headers)
                            if status_resp.status_code == 200:
                                status_data = status_resp.json()
                                status = status_data.get("status")
                                print(f"[Firecrawl Debug] Crawl status: {status}")
                                
                                if status == "completed":
                                    pages = status_data.get("data", [])
                                    print(f"[Firecrawl Debug] Crawl completed with {len(pages)} pages")
                                    
                                    for page in pages:
                                        content = page.get("content") or page.get("markdown", "")
                                        if content and len(content.split()) >= min_words:
                                            source_url = page.get("metadata", {}).get("sourceURL", url)
                                            urls_crawled.append(source_url)
                                            all_documents.append(Document(page_content=content, metadata={"source": source_url}))
                                    break
                                elif status == "failed":
                                    print(f"[Firecrawl Debug] Crawl failed")
                                    break
                                else:
                                    time.sleep(2)
                            else:
                                print(f"[Firecrawl Debug] Error checking crawl status: {status_resp.status_code}")
                                break
                    else:
                        print(f"[Firecrawl Debug] No job ID returned")
                else:
                    print(f"[Firecrawl Debug] Error starting crawl: {resp.status_code} - {resp.text}")
            else:
                # Use FirecrawlApp SDK
                if not FirecrawlApp:
                    print("[Firecrawl Debug] FirecrawlApp not available and no session_cookie provided")
                    return ([], {"pages_crawled": 0, "urls_crawled": []}) if return_analytics else []
                
                app = FirecrawlApp(api_key=api_key)
                crawl_result = app.crawl_url(url, params={"limit": limit, "formats": formats})
                
                if crawl_result and 'data' in crawl_result:
                    for page in crawl_result['data']:
                        content = page.get('content') or page.get('markdown', '')
                        if content and len(content.split()) >= min_words:
                            source_url = page.get('metadata', {}).get('sourceURL', url)
                            urls_crawled.append(source_url)
                            all_documents.append(Document(page_content=content, metadata={"source": source_url}))
                            
        except Exception as e:
            print(f"[Firecrawl Debug] Deep crawl error: {e}")
    else:
        # Shallow crawl for multiple URLs or single page scraping
        for url in urls:
            try:
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
                        "formats": formats
                    }
                    
                    resp = requests.post("https://api.firecrawl.dev/v1/scrape", headers=headers, json=data)
                    
                    if resp.status_code == 200:
                        result = resp.json()
                        if result.get("success"):
                            page_data = result.get("data", {})
                            content = page_data.get("content") or page_data.get("markdown", "")
                            if content and len(content.split()) >= min_words:
                                urls_crawled.append(url)
                                all_documents.append(Document(page_content=content, metadata={"source": url}))
                        else:
                            print(f"[Firecrawl Debug] Scrape failed for {url}: {result.get('error')}")
                    else:
                        print(f"[Firecrawl Debug] Error scraping {url}: {resp.status_code} - {resp.text}")
                else:
                    # Use FirecrawlApp SDK
                    if not FirecrawlApp:
                        print(f"[Firecrawl Debug] FirecrawlApp not available and no session_cookie provided for {url}")
                        continue
                    
                    app = FirecrawlApp(api_key=api_key)
                    scrape_result = app.scrape_url(url, params={"formats": formats})
                    
                    if scrape_result and scrape_result.get('success'):
                        content = scrape_result['data'].get('content') or scrape_result['data'].get('markdown', '')
                        if content and len(content.split()) >= min_words:
                            urls_crawled.append(url)
                            all_documents.append(Document(page_content=content, metadata={"source": url}))
                    else:
                        print(f"[Firecrawl Debug] Scrape failed for {url}: {scrape_result.get('error') if scrape_result else 'Unknown error'}")
                        
            except Exception as e:
                print(f"[Firecrawl Debug] Error scraping {url}: {e}")
    
    analytics = {
        "pages_crawled": len(all_documents),
        "urls_crawled": urls_crawled
    }
    
    print(f"[Firecrawl Debug] Total documents extracted: {len(all_documents)}")
    
    if return_analytics:
        return all_documents, analytics
    return all_documents

def get_text_splitter():
    """
    Returns a RecursiveCharacterTextSplitter for document chunking.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

def aggregate_crawl_analytics(website_analytics, link_analytics, file_analytics):
    """
    Aggregate analytics from website crawling, link processing, and file processing.
    """
    analytics = {
        "pages_crawled": 0,
        "urls_crawled": [],
        "files_indexed": 0
    }
    
    if website_analytics:
        analytics["pages_crawled"] += website_analytics.get("pages_crawled", 0)
        analytics["urls_crawled"].extend(website_analytics.get("urls_crawled", []))
    
    if link_analytics:
        analytics["pages_crawled"] += link_analytics.get("pages_crawled", 0)
        analytics["urls_crawled"].extend(link_analytics.get("urls_crawled", []))
    
    if file_analytics:
        analytics["files_indexed"] = file_analytics.get("files_indexed", 0)
    
    return analytics

def generate_prompt_template(company_info):
    """
    Generate a basic prompt template for the agent.
    """
    return f"""
You are {company_info.get('bot_name', 'AI Assistant')} for {company_info.get('company_name', 'the company')}.

Use the following context to answer the user's question:
{{context}}

Question: {{question}}
Answer:
"""

def create_chat_prompt_template(company_info):
    """
    Create a ChatPromptTemplate for the conversational chain.
    """
    template = generate_prompt_template(company_info)
    return ChatPromptTemplate.from_template(template)

def create_conversational_retrieval_chain(llm, retriever, prompt_template, memory):
    """
    Create a conversational retrieval chain.
    """
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=False
    )