from flask import Flask, request, jsonify
from rag_dynamic import DynamicRAGAgent
# Import supabase client and credentials from the centralized module
from supabase_client import supabase, SUPABASE_URL, SUPABASE_KEY
import tempfile
import requests
from langchain.docstore.document import Document
from rag_utils import (
    delete_vectors_by_url,
    get_embeddings,
    get_text_splitter,
    download_faiss_index_from_supabase,
    upload_faiss_index_to_supabase,
    append_to_vectorstore,
    extract_website_text_with_firecrawl,
    generic_create_vectorstore,
    extract_file_text
)
# LangChain deprecation warning: Use from langchain_community.vectorstores instead
from langchain.vectorstores import FAISS
import traceback
import os
import uuid

# Storage bucket name for vectorstores
SUPABASE_STORAGE_BUCKET = "vectorstores"

app = Flask(__name__)

# Enable CORS for the CRM frontend
from flask_cors import CORS
CORS(app, origins=["https://crm.growbro.ai"])

@app.route("/trigger", methods=["POST"])
def trigger_vectorstore():
    data = request.json
    # Supabase sends the updated row as 'record'
    record = data.get("record", {})
    ai_id = record.get("id")
    vectorstore_ready = record.get("vectorstore_ready")
    if not ai_id:
        return jsonify({"status": "error", "message": "Missing ai_id"}), 400

    # Only trigger if vectorstore_ready is False
    if vectorstore_ready is not False:
        return jsonify({"status": "ignored", "message": "No action needed"}), 200

    session_cookie = record.get("session_cookie")
    try:
        agent = DynamicRAGAgent(ai_id, session_cookie=session_cookie)
        agent.extract_and_build_vectorstore(force_rebuild=True)  # Always rebuild!
        if agent.is_ready():
            supabase.table("business_info").update({"vectorstore_ready": True}).eq("id", ai_id).execute()
            return jsonify({"status": "success", "message": f"Vectorstore rebuilt for {ai_id}"}), 200
        else:
            return jsonify({"status": "error", "message": "Vectorstore not ready after build"}), 500
    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500

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
        
        # Download FAISS index from Supabase if available
        vectorstore_path = f"faiss_index_{ai_id}"
        faiss_index_file = os.path.join(vectorstore_path, "index.faiss")
        if (not os.path.exists(faiss_index_file)) and SUPABASE_URL and SUPABASE_KEY:
            try:
                download_faiss_index_from_supabase(ai_id, SUPABASE_URL, SUPABASE_STORAGE_BUCKET, local_dir=".")
                print(f"[add_files] Downloaded vectorstore from Supabase")
            except Exception as e:
                print(f"[add_files] Warning: Could not download FAISS index from Supabase: {e}")
                return jsonify({"status": "error", "message": "Vectorstore not found and could not be downloaded"}), 404
                
        try:
            # Load existing vectorstore
            embeddings = get_embeddings()
            text_splitter = get_text_splitter()
            vectorstore = load_faiss_vectorstore(vectorstore_path, embeddings)
            print(f"[add_files] Loaded existing vectorstore from {vectorstore_path}")
            
            # Process new files
            new_docs = []
            files_processed = []
            
            # Add skipped files to analytics
            for url in file_urls:
                if url in existing_files:
                    file_stats["file_status"][url] = "already_processed"
            
            # Process files similar to extract_and_build_vectorstore
            for file_url in new_file_urls:
                try:
                    resp = requests.get(file_url)
                    resp.raise_for_status()
                    # Extract file extension
                    import os
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
            
            # Append new documents to the existing vectorstore
            print(f"[add_files] Appending {len(new_docs)} new documents to vectorstore")
            updated_vectorstore = append_to_vectorstore(vectorstore, new_docs, embeddings, text_splitter)
            
            # Save the vectorstore directly to the output directory
            os.makedirs(vectorstore_path, exist_ok=True)
            updated_vectorstore.save_local(vectorstore_path)
            print(f"[add_files] Saved updated FAISS index to: {vectorstore_path}")
            
            # Upload the updated index to Supabase
            print(f"[add_files] Uploading updated vectorstore to Supabase")
            upload_faiss_index_to_supabase(
                ai_id,
                supabase_url=SUPABASE_URL,
                bucket=SUPABASE_STORAGE_BUCKET,
                supabase_key=SUPABASE_KEY,
                local_dir="."  # Use current directory as base
            )
            
            # Update files_indexed count in business_info
            try:
                # First get current business_info to preserve other analytics
                business_info_res = supabase.table("business_info").select("*").eq("id", ai_id).single().execute()
                business_info = business_info_res.data
                
                if business_info:
                    # Get current count or default to 0
                    current_files_indexed = business_info.get("files_indexed", 0)
                    
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
            print(f"[add_files] Error processing vectorstore: {str(e)}")
            traceback.print_exc()
            return jsonify({'status': 'error', 'message': str(e)}), 500
            
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
        
        # Download existing vectorstore
        print(f"[add_links] Downloading existing vectorstore for {ai_id}")
        try:
            # Set up embeddings and text splitter
            embeddings = get_embeddings()
            text_splitter = get_text_splitter()

            # Download the existing vectorstore
            local_index_dir = download_faiss_index_from_supabase(
                ai_id,
                supabase_url=SUPABASE_URL,
                bucket=SUPABASE_STORAGE_BUCKET
            )
            vectorstore = FAISS.load_local(local_index_dir, embeddings, allow_dangerous_deserialization=True)
            print(f"[add_links] Successfully loaded existing FAISS index with {len(vectorstore.docstore._dict)} vectors")
            
            # Crawl only the new URLs - no deep crawl, just the specific new URLs
            print(f"[add_links] Crawling {len(new_urls_to_crawl)} new URLs for {ai_id}")
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
            
            # Append new documents to the existing vectorstore
            print(f"[add_links] Appending {len(new_docs)} new documents to vectorstore")
            updated_vectorstore = append_to_vectorstore(vectorstore, new_docs, embeddings, text_splitter)
            
            # Save the vectorstore directly to the output directory
            output_dir = f"faiss_index_{ai_id}"
            os.makedirs(output_dir, exist_ok=True)
            updated_vectorstore.save_local(output_dir)
            
            # Create version.txt file
            import datetime
            version_txt_path = os.path.join(output_dir, "version.txt")
            version = datetime.datetime.utcnow().isoformat()
            with open(version_txt_path, "w") as f:
                f.write(version)
                
            print(f"[add_links] Saved updated FAISS index to: {output_dir}")
            
            # Upload the updated index to Supabase
            print(f"[add_links] Uploading updated vectorstore to Supabase")
            upload_faiss_index_to_supabase(
                ai_id,
                supabase_url=SUPABASE_URL,
                bucket=SUPABASE_STORAGE_BUCKET,
                supabase_key=SUPABASE_KEY,
                local_dir="."  # Use current directory as base, not the faiss_index dir itself
            )
            
            # Update business_info with new urls_crawled list and total_pages_crawled
            # Use the actually crawled URLs from analytics, not just the requested ones
            print(f"[add_links] Updating urls_crawled and total_pages_crawled in DB for {ai_id}")
            new_urls_list = list(set(existing_urls + actually_crawled_urls))  # Remove duplicates, use actually crawled URLs
            res = supabase.table("business_info").update({
                "urls_crawled": new_urls_list,
                "total_pages_crawled": len(new_urls_list)  # Keep total_pages_crawled in sync
            }).eq("id", ai_id).execute()
            
            return jsonify({
                'status': 'success',
                'added_count': len(new_docs),
                'urls_crawled': new_urls_list,
                'actually_crawled': actually_crawled_urls,
                'pages_crawled': analytics['pages_crawled']
            })
            
        except Exception as e:
            print(f"[add_links] Error processing vectorstore: {str(e)}")
            traceback.print_exc()
            return jsonify({'status': 'error', 'message': str(e)}), 500
            
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

    # 3. Delete vectors (always download latest from Supabase Storage first)
    import os
    from rag_utils import get_embeddings, load_faiss_vectorstore, delete_vectors_by_url, download_faiss_index_from_supabase, upload_faiss_index_to_supabase
    from supabase_client import SUPABASE_URL, SUPABASE_KEY
    SUPABASE_BUCKET = "vectorstores"
    vectorstore_path = f"faiss_index_{ai_id}"
    faiss_index_file = os.path.join(vectorstore_path, "index.faiss")
    print(f"[remove_urls] Looking for FAISS index at: {faiss_index_file}")
    # Download latest vectorstore if not present locally
    if (not os.path.exists(faiss_index_file)) and SUPABASE_URL and SUPABASE_KEY:
        try:
            download_faiss_index_from_supabase(ai_id, SUPABASE_URL, SUPABASE_BUCKET, local_dir=".")
        except Exception as e:
            print(f"[remove_urls] Warning: Could not download FAISS index from Supabase: {e}")
    deleted_count = 0
    if os.path.exists(faiss_index_file):
        embeddings = get_embeddings()
        vectorstore = load_faiss_vectorstore(vectorstore_path, embeddings)
        deleted_count = delete_vectors_by_url(vectorstore, urls_to_remove)
        print(f"[remove_urls] delete_vectors_by_url returned: {deleted_count}")
        vectorstore.save_local(vectorstore_path)
        print(f"[remove_urls] Saved updated FAISS index to: {vectorstore_path}")
        # Print existence of all FAISS files before upload
        import os
        for fname in ["index.faiss", "index.pkl", "splits.pkl"]:
            fpath = os.path.join(vectorstore_path, fname)
            print(f"[remove_urls] File {fpath} exists? {os.path.exists(fpath)}")
        # Upload updated vectorstore to Supabase Storage
        try:
            print(f"[remove_urls] Uploading FAISS index from directory: {vectorstore_path}")
            upload_faiss_index_to_supabase(ai_id, SUPABASE_URL, SUPABASE_BUCKET, SUPABASE_KEY, local_dir=".")
        except Exception as e:
            print(f"[remove_urls] Warning: Could not upload FAISS index to Supabase: {e}")

        # 4. Update DB (only after upload is done)
        supabase.table("business_info").update({
            "urls_crawled": new_urls,
            "total_pages_crawled": len(new_urls)  # Keep total_pages_crawled in sync with urls_crawled
        }).eq("id", ai_id).execute()
        print(f"[remove_urls] Updated urls_crawled and total_pages_crawled in DB for {ai_id} after FAISS upload.")

        response = {"status": "success", "deleted_count": deleted_count, "new_urls": new_urls}
        print(f"[remove_urls] Returning response: {response}")
        return jsonify(response), 200


@app.route("/remove-files", methods=["POST"])
def remove_files():
    data = request.json
    ai_id = data.get("ai_id")
    file_urls = data.get("file_urls", [])
    if not ai_id or not file_urls:
        return jsonify({"status": "error", "message": "Missing ai_id or file_urls"}), 400

    try:
        print(f"[remove_files] Removing {len(file_urls)} files from vectorstore for AI {ai_id}")
        
        # 1. Download the existing vectorstore from Supabase
        print(f"[remove_files] Downloading vectorstore from Supabase")
        vectorstore_path = f"faiss_index_{ai_id}"
        faiss_index_file = os.path.join(vectorstore_path, "index.faiss")
        print(f"[remove_files] Looking for FAISS index at: {faiss_index_file}")
        
        # Download latest vectorstore if not present locally
        if (not os.path.exists(faiss_index_file)) and SUPABASE_URL and SUPABASE_KEY:
            try:
                print(f"[remove_files] Downloading vectorstore from Supabase")
                download_faiss_index_from_supabase(ai_id, SUPABASE_URL, SUPABASE_STORAGE_BUCKET, local_dir=".")
                print(f"[remove_files] Successfully downloaded vectorstore")
            except Exception as e:
                print(f"[remove_files] Error downloading vectorstore: {e}")
                return jsonify({"status": "error", "message": f"Error downloading vectorstore: {str(e)}"}), 500
        
        # 2. Check if vectorstore exists locally
        if not os.path.exists(vectorstore_path) or not os.path.isdir(vectorstore_path):
            print(f"[remove_files] Error: Vectorstore directory not found at {vectorstore_path}")
            return jsonify({"status": "error", "message": f"Vectorstore not found for AI {ai_id}"}), 404
        
        # 3. Load the existing vectorstore
        print(f"[remove_files] Loading vectorstore from {vectorstore_path}")
        try:
            embeddings = get_embeddings()
            vectorstore = FAISS.load_local(vectorstore_path, embeddings)
            print(f"[remove_files] Successfully loaded vectorstore")
        except Exception as e:
            print(f"[remove_files] Error loading vectorstore: {e}")
            return jsonify({"status": "error", "message": f"Error loading vectorstore: {str(e)}"}), 500
        
        # 4. Delete vectors for each file URL
        deleted_count = 0
        for file_url in file_urls:
            try:
                # The source in document metadata should match the file URL
                print(f"[remove_files] Deleting vectors for file: {file_url}")
                deleted = delete_vectors_by_url(vectorstore, file_url)
                deleted_count += deleted
                print(f"[remove_files] Deleted {deleted} vectors for file {file_url}")
            except Exception as e:
                print(f"[remove_files] Error deleting vectors for file {file_url}: {e}")
        
        # 5. Save the updated vectorstore if any vectors were deleted
        if deleted_count > 0:
            print(f"[remove_files] Saving updated vectorstore to {vectorstore_path}")
            vectorstore.save_local(vectorstore_path)
            
            # 6. Upload the updated index to Supabase
            print(f"[remove_files] Uploading updated vectorstore to Supabase")
            upload_faiss_index_to_supabase(
                ai_id,
                supabase_url=SUPABASE_URL,
                bucket=SUPABASE_STORAGE_BUCKET,
                supabase_key=SUPABASE_KEY,
                local_dir="."  # Use current directory as base
            )
            
            # 7. Update files_indexed count in business_info
            try:
                # First get current business_info to preserve other analytics
                business_info_res = supabase.table("business_info").select("*").eq("id", ai_id).single().execute()
                business_info = business_info_res.data
                
                if business_info:
                    # Get current count or default to 0
                    current_files_indexed = business_info.get("files_indexed", 0)
                    
                    # Calculate new count (subtract deleted files)
                    new_files_indexed = max(0, current_files_indexed - len(file_urls))
                    
                    # Update business_info with new count
                    print(f"[remove_files] Updating files_indexed count from {current_files_indexed} to {new_files_indexed}")
                    supabase.table("business_info").update({"files_indexed": new_files_indexed}).eq("id", ai_id).execute()
            except Exception as e:
                print(f"[remove_files] Warning: Could not update files_indexed count: {e}")
        
        # 8. Return response
        response = {
            'status': 'success',
            'deleted_count': deleted_count,
            'files_processed': len(file_urls),
            'message': f'Successfully removed {deleted_count} vectors from the vectorstore'
        }
        print(f"[remove_files] Returning response: {response}")
        return jsonify(response), 200
    
    except Exception as e:
        print(f"[remove_files] Error: {str(e)}")
        traceback.print_exc()
        return jsonify({"status": "error", "message": f"Error removing files: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)

