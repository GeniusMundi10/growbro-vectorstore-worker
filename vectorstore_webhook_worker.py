from flask import Flask, request, jsonify
from rag_dynamic import DynamicRAGAgent
# Import supabase client and credentials from the centralized module
from supabase_client import supabase, SUPABASE_URL, SUPABASE_KEY
from rag_utils import (
    delete_vectors_by_url,
    get_embeddings,
    get_text_splitter,
    download_faiss_index_from_supabase,
    upload_faiss_index_to_supabase,
    append_to_vectorstore,
    extract_website_text_with_firecrawl
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
            
            # Save the updated vectorstore
            output_dir = f"faiss_index_{ai_id}"
            os.makedirs(output_dir, exist_ok=True)
            updated_vectorstore.save_local(output_dir)
            print(f"[add_links] Saved updated FAISS index to: {output_dir}")
            
            # Upload the updated index to Supabase
            print(f"[add_links] Uploading updated vectorstore to Supabase")
            upload_faiss_index_to_supabase(
                ai_id,
                supabase_url=SUPABASE_URL,
                bucket=SUPABASE_STORAGE_BUCKET,
                supabase_key=SUPABASE_KEY,
                local_dir=output_dir
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)

