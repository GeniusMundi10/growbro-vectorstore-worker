from flask import Flask, request, jsonify
from rag_dynamic import DynamicRAGAgent
# Import supabase client and credentials from the centralized module
from supabase_client import supabase, SUPABASE_URL, SUPABASE_KEY
import tempfile
import requests
from langchain_core.documents import Document
from rag_utils import (
    extract_website_text_with_firecrawl,
    extract_file_text
)
from pinecone_serverless_utils import (
    check_index_exists,
    check_namespace_exists,
    upsert_documents_with_lightweight_embeddings,
    append_documents_to_pinecone,
    delete_vectors_by_ai_id,
    delete_vectors_by_source,
    get_vectorstore_stats,
    
)
import traceback
import os
import uuid

# Storage bucket name for vectorstores
SUPABASE_STORAGE_BUCKET = "vectorstores"

app = Flask(__name__)

# Enable CORS for the CRM frontend
from flask_cors import CORS
CORS(app, origins=["https://crm.growbro.ai"])

PROCESSING_IDS = set()

@app.route("/create-vectorstore", methods=["POST"])
def create_vectorstore():
    """
    Endpoint to create and build a new vectorstore for a user. Should be called only on user signup.
    Expects JSON body with:
      - ai_id (required): The AI/business ID for which to create the vectorstore
      - session_cookie (optional): If needed for downstream auth
    """
    data = request.json
    ai_id = data.get("ai_id")
    session_cookie = data.get("session_cookie")
    if not ai_id:
        return jsonify({"status": "error", "message": "Missing ai_id"}), 400

    # In-memory lock to prevent duplicate builds for same ai_id
    if ai_id in PROCESSING_IDS:
        return jsonify({"status": "ignored", "message": "Build already in progress"}), 200
    PROCESSING_IDS.add(ai_id)

    try:
        agent = DynamicRAGAgent(ai_id, session_cookie=session_cookie)
        agent.extract_and_build_vectorstore(force_rebuild=True)
        if agent.is_ready():
            supabase.table("business_info").update({"vectorstore_ready": True}).eq("id", ai_id).execute()
            return jsonify({"status": "success", "message": f"Vectorstore created for {ai_id}"}), 200
        else:
            return jsonify({"status": "error", "message": "Vectorstore not ready after build"}), 500
    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        PROCESSING_IDS.discard(ai_id)

@app.route("/add-files", methods=["POST"])
def add_files():
    """
    Add new files to an existing vectorstore without triggering a full rebuild.
    This endpoint allows for incremental file additions to the knowledge base.
    
    Required params:
    - ai_id: The AI ID to add files to
    - file_urls: List of file URLs from Supabase Storage to process
    
    Returns:
    - Success status and analytics about processed files
    """
    # Import os at function scope to avoid UnboundLocalError
    import os
    
    try:
        data = request.json
        ai_id = data.get("ai_id")
        file_urls = data.get("file_urls", [])
        
        if not ai_id or not file_urls:
            return jsonify({"status": "error", "message": "Missing ai_id or file_urls"}), 400
            
        print(f"[add_files] Processing {len(file_urls)} new files for AI {ai_id}")
        
        # Fetch existing files from DB to avoid duplicates
        res = supabase.table("ai_file").select("url").eq("ai_id", ai_id).execute()
        existing_files = [item.get("url") for item in res.data] if res.data else []
        
        # Filter out any files that are already in the vectorstore
        new_file_urls = [url for url in file_urls if url not in existing_files]
        
        # Prepare analytics
        file_stats = {
            "total_requested": len(file_urls),
            "already_processed": len(file_urls) - len(new_file_urls),
            "to_process": len(new_file_urls),
            "successfully_processed": 0,
            "failed": 0,
            "file_status": {}
        }
        
        # If no new files to process
        if not new_file_urls:
            print(f"[add_files] All files already in vectorstore")
            return jsonify({
                "status": "success", 
                "message": "All files already processed",
                "analytics": file_stats
            }), 200
        
        print(f"[add_files] Found {len(new_file_urls)} new files to process")
        
        # Check if consolidated Pinecone index exists
        if not check_index_exists():
            print(f"[add_files] No consolidated Pinecone index found. Will be created automatically.")
        else:
            print(f"[add_files] Found consolidated Pinecone index. Will check namespace.")
            
        # Check if this AI's namespace exists
        if not check_namespace_exists(ai_id):
            print(f"[add_files] No namespace found for AI {ai_id}. Will be created automatically.")
        else:
            print(f"[add_files] Found existing namespace for AI {ai_id}")
            
        # Process new files
        new_docs = []
        files_processed = []
        
        # Add skipped files to analytics
        for url in file_urls:
            if url in existing_files:
                file_stats["file_status"][url] = "already_processed"
        
        # Process each new file
        for file_url in new_file_urls:
            try:
                resp = requests.get(file_url)
                resp.raise_for_status()
                # Extract file extension
                suffix = os.path.splitext(file_url)[-1]
                
                # Process with temp file
                with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp_file:
                    tmp_file.write(resp.content)
                    tmp_file.flush()
                    text = extract_file_text(tmp_file.name)
                    if text:
                        new_docs.append(Document(page_content=text, metadata={"source": file_url}))
                        files_processed.append(file_url)
                        file_stats["file_status"][file_url] = "success"
                        file_stats["successfully_processed"] += 1
                        print(f"[add_files] Successfully extracted text from {file_url}")
                    else:
                        file_stats["file_status"][file_url] = "no_content_extracted"
                        file_stats["failed"] += 1
                        print(f"[add_files] No text extracted from file: {file_url}")
            except Exception as e:
                file_stats["file_status"][file_url] = f"error: {str(e)}"
                file_stats["failed"] += 1
                print(f"[add_files] Error downloading or extracting file: {file_url}, error: {e}")
        
        # If no documents were extracted
        if not new_docs:
            print(f"[add_files] No content extracted from files")
            return jsonify({
                'status': 'error', 
                'message': 'Failed to extract content from files',
                'analytics': file_stats
            }), 400
        
        # Add new documents to Pinecone index
        print(f"[add_files] Adding {len(new_docs)} new documents to Pinecone index")
        try:
            append_documents_to_pinecone(ai_id, new_docs)
            print(f"[add_files] Successfully added documents to Pinecone index")
        except Exception as e:
            print(f"[add_files] Error adding documents to Pinecone: {e}")
            return jsonify({
                'status': 'error',
                'message': f'Error adding documents to Pinecone: {str(e)}'
            }), 500
        
        # Update files_indexed count in business_info
        try:
            # First get current business_info to preserve other analytics
            business_info_res = supabase.table("business_info").select("*").eq("id", ai_id).single().execute()
            business_info = business_info_res.data
            
            if business_info:
                # Get current count or default to 0 if None
                current_files_indexed = business_info.get("files_indexed") or 0
                
                # Calculate new count (add newly processed files)
                new_files_indexed = current_files_indexed + file_stats["successfully_processed"]
                
                # Update business_info with new count
                print(f"[add_files] Updating files_indexed count from {current_files_indexed} to {new_files_indexed}")
                supabase.table("business_info").update({"files_indexed": new_files_indexed}).eq("id", ai_id).execute()
                
                # Update analytics for response
                file_stats["total_files_indexed"] = new_files_indexed
        except Exception as e:
            print(f"[add_files] Warning: Could not update files_indexed count: {e}")
        
        return jsonify({
            'status': 'success',
            'added_count': len(new_docs),
            'files_processed': files_processed,
            'analytics': file_stats
        })
            
    except Exception as e:
        print(f"[add_files] Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/add-links', methods=['POST'])
def add_links():
    # Get the ai_id and new URLs from the request
    data = request.get_json()
    ai_id = data.get('ai_id')
    new_urls = data.get('new_urls', [])
    
    if not ai_id or not new_urls:
        return jsonify({
            'status': 'error', 
            'message': 'Missing required parameters: ai_id and/or new_urls'
        }), 400
    
    print(f"[add_links] Adding {len(new_urls)} new URLs for {ai_id}")
    
    # Get business info from Supabase to get existing URLs
    try:
        print(f"[add_links] Fetching business info for {ai_id}")
        res = supabase.table("business_info").select("*").eq("id", ai_id).single().execute()
        business_info = res.data
        if not business_info:
            print(f"[add_links] No business info found for {ai_id}")
            return jsonify({'status': 'error', 'message': 'Business info not found'}), 404
            
        # Get current crawled URLs
        existing_urls = business_info.get("urls_crawled", [])        
        # Check if any of the new URLs are already in the vectorstore
        already_crawled = set(existing_urls).intersection(set(new_urls))
        new_urls_to_crawl = [url for url in new_urls if url not in already_crawled]
        
        if not new_urls_to_crawl:
            print(f"[add_links] All URLs already in vectorstore, nothing to add")
            return jsonify({
                'status': 'success', 
                'added_count': 0,
                'urls_crawled': existing_urls
            })
        
        # Check if consolidated Pinecone index exists
        print(f"[add_links] Checking Pinecone index for {ai_id}")
        if not check_index_exists():
            print(f"[add_links] No consolidated Pinecone index found. Will be created automatically.")
            
        # Check if this AI's namespace exists
        if not check_namespace_exists(ai_id):
            print(f"[add_links] No namespace found for AI {ai_id}. Will be created automatically.")
        else:
            print(f"[add_links] Found existing namespace for AI {ai_id}")
        
        # Crawl the new URLs
        print(f"[add_links] Crawling {len(new_urls_to_crawl)} new URLs for {ai_id}")
        try:
            new_docs, analytics = extract_website_text_with_firecrawl(
                new_urls_to_crawl,
                return_analytics=True, 
                deep_crawl=False  # No deep crawl, just the specific URLs
            )
            
            # Use the actual crawled URLs from analytics, not the requested URLs
            actually_crawled_urls = analytics["urls_crawled"]
            print(f"[add_links] Successfully crawled {len(actually_crawled_urls)} URLs: {actually_crawled_urls}")
            
            if not new_docs:
                print(f"[add_links] No documents extracted from new URLs")
                return jsonify({
                    'status': 'error', 
                    'message': 'Failed to extract content from new URLs'
                }), 400
            
            # Add new documents to Pinecone index
            print(f"[add_links] Adding {len(new_docs)} new documents to Pinecone index")
            append_documents_to_pinecone(ai_id, new_docs)
            print(f"[add_links] Successfully added documents to Pinecone index")
            
            # Update business_info with new urls_crawled list and total_pages_crawled
            print(f"[add_links] Updating urls_crawled and total_pages_crawled in DB for {ai_id}")
            new_urls_list = list(set(existing_urls + actually_crawled_urls))  # Remove duplicates
            supabase.table("business_info").update({
                "urls_crawled": new_urls_list,
                "total_pages_crawled": len(new_urls_list)
            }).eq("id", ai_id).execute()
            
            return jsonify({
                'status': 'success',
                'added_count': len(new_docs),
                'urls_crawled': new_urls_list,
                'actually_crawled': actually_crawled_urls,
                'pages_crawled': analytics['pages_crawled']
            })
            
        except Exception as e:
            print(f"[add_links] Error adding documents to Pinecone: {str(e)}")
            traceback.print_exc()
            return jsonify({
                'status': 'error',
                'message': f'Error adding documents to Pinecone: {str(e)}'
            }), 500
            
    except Exception as e:
        print(f"[add_links] Error: {str(e)}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route("/remove-urls", methods=["POST"])
def remove_urls():
    data = request.json
    ai_id = data.get("ai_id")
    urls_to_remove = data.get("urls_to_remove", [])
    if not ai_id or not urls_to_remove:
        return jsonify({"status": "error", "message": "Missing ai_id or urls_to_remove"}), 400

    # 1. Fetch current urls_crawled
    res = supabase.table("business_info").select("urls_crawled").eq("id", ai_id).execute()
    if not res.data or not isinstance(res.data, list):
        return jsonify({"status": "error", "message": "AI not found"}), 404
    current_urls = res.data[0].get("urls_crawled", [])
    print(f"[remove_urls] Current urls_crawled from DB: {current_urls}")
    if isinstance(current_urls, str):
        import json
        current_urls = json.loads(current_urls)
    if not isinstance(current_urls, list):
        current_urls = []

    # 2. Compute new_urls
    new_urls = [u for u in current_urls if u not in urls_to_remove]
    print(f"[remove_urls] new_urls after removal: {new_urls}")

    # 3. Delete vectors from Pinecone namespace
    print(f"[remove_urls] Deleting vectors for URLs: {urls_to_remove}")
    try:
        deleted_count = delete_vectors_by_source(ai_id, urls_to_remove)
        print(f"[remove_urls] Successfully deleted {deleted_count} vectors from Pinecone")

        # 4. Update DB after successful deletion
        supabase.table("business_info").update({
            "urls_crawled": new_urls,
            "total_pages_crawled": len(new_urls)
        }).eq("id", ai_id).execute()
        print(f"[remove_urls] Updated urls_crawled and total_pages_crawled in DB for {ai_id}")

        response = {"status": "success", "deleted_count": deleted_count, "new_urls": new_urls}
        print(f"[remove_urls] Returning response: {response}")
        return jsonify(response), 200
        
    except Exception as e:
        print(f"[remove_urls] Error deleting from Pinecone: {str(e)}")
        traceback.print_exc()
        return jsonify({
            "status": "error", 
            "message": f"Error deleting vectors: {str(e)}"
        }), 500


@app.route("/remove-files", methods=["POST"])
def remove_files():
    data = request.json
    ai_id = data.get("ai_id")
    file_urls = data.get("file_urls", [])
    if not ai_id or not file_urls:
        return jsonify({"status": "error", "message": "Missing ai_id or file_urls"}), 400

    try:
        print(f"[remove_files] Removing {len(file_urls)} files from Pinecone index for AI {ai_id}")
        
        # Delete vectors for each file URL from Pinecone
        deleted_count = 0
        for file_url in file_urls:
            try:
                print(f"[remove_files] Deleting vectors for file: {file_url}")
                deleted = delete_vectors_by_source(ai_id, [file_url])
                deleted_count += deleted
                print(f"[remove_files] Deleted {deleted} vectors for file {file_url}")
            except Exception as e:
                print(f"[remove_files] Error deleting vectors for file {file_url}: {e}")
        
        # Update files_indexed count in business_info
        try:
            # First get current business_info to preserve other analytics
            business_info_res = supabase.table("business_info").select("*").eq("id", ai_id).single().execute()
            business_info = business_info_res.data
            
            if business_info:
                # Get current count or default to 0 if None
                current_files_indexed = business_info.get("files_indexed") or 0
                
                # Calculate new count (subtract deleted files)
                new_files_indexed = max(0, current_files_indexed - len(file_urls))
                
                # Update business_info with new count
                print(f"[remove_files] Updating files_indexed count from {current_files_indexed} to {new_files_indexed}")
                supabase.table("business_info").update({"files_indexed": new_files_indexed}).eq("id", ai_id).execute()
        except Exception as e:
            print(f"[remove_files] Warning: Could not update files_indexed count: {e}")
        
        # Return response
        response = {
            'status': 'success',
            'deleted_count': deleted_count,
            'files_processed': len(file_urls),
            'message': f'Successfully removed {deleted_count} vectors from Pinecone index'
        }
        print(f"[remove_files] Returning response: {response}")
        return jsonify(response), 200
    
    except Exception as e:
        print(f"[remove_files] Error: {str(e)}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Error removing files: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)

