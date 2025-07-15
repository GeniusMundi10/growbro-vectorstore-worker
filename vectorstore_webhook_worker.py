from flask import Flask, request, jsonify
from rag_dynamic import DynamicRAGAgent
from supabase_client import supabase
from rag_utils import delete_vectors_by_url
import traceback

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
    if isinstance(current_urls, str):
        import json
        current_urls = json.loads(current_urls)
    if not isinstance(current_urls, list):
        current_urls = []

    # 2. Compute new_urls
    new_urls = [u for u in current_urls if u not in urls_to_remove]

    # 3. Delete vectors (always download latest from Supabase Storage first)
    import os
    from rag_utils import get_embeddings, load_faiss_vectorstore, delete_vectors_by_url, download_faiss_index_from_supabase, upload_faiss_index_to_supabase
    from supabase_client import SUPABASE_URL, SUPABASE_KEY
    SUPABASE_BUCKET = "vectorstores"
    vectorstore_path = f"faiss_index_{ai_id}"
    faiss_index_file = os.path.join(vectorstore_path, "index.faiss")
    # Download latest vectorstore if not present locally
    if (not os.path.exists(faiss_index_file)) and SUPABASE_URL and SUPABASE_KEY:
        try:
            download_faiss_index_from_supabase(ai_id, SUPABASE_URL, SUPABASE_BUCKET)
        except Exception as e:
            print(f"[remove_urls] Warning: Could not download FAISS index from Supabase: {e}")
    deleted_count = 0
    if os.path.exists(faiss_index_file):
        embeddings = get_embeddings()
        vectorstore = load_faiss_vectorstore(vectorstore_path, embeddings)
        deleted_count = delete_vectors_by_url(vectorstore, urls_to_remove)
        vectorstore.save_local(vectorstore_path)
        # Upload updated vectorstore to Supabase Storage
        try:
            upload_faiss_index_to_supabase(ai_id, SUPABASE_URL, SUPABASE_BUCKET, SUPABASE_KEY, local_dir=vectorstore_path)
        except Exception as e:
            print(f"[remove_urls] Warning: Could not upload FAISS index to Supabase: {e}")

    # 4. Update DB
    supabase.table("business_info").update({
        "urls_crawled": new_urls
    }).eq("id", ai_id).execute()

    return jsonify({"status": "success", "deleted_count": deleted_count, "new_urls": new_urls}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)

