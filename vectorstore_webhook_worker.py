from flask import Flask, request, jsonify
from rag_dynamic import DynamicRAGAgent
from supabase_client import supabase
import traceback

app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001)
