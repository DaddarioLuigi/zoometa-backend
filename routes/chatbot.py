# routes/chatbot.py
from flask import Blueprint, request, jsonify, send_file
from services.ChatService import ChatService
from models import db, Conversation, Reviews
import uuid
import redis
import json
import os
import tempfile
from gtts import gTTS
import openai

chatbot_bp = Blueprint('chatbot', __name__)

# Inizializzazione di Redis (se non già inizializzato in app.py)
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
chat_services = {}

def create_session():
    session_id = str(uuid.uuid4())
    chat_service = ChatService(
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        kb_dir="/Users/luigidaddario/Downloads/kb_zoometa_arianna/kb",
        product_kb_dir="/Users/luigidaddario/Downloads/kb_zoometa_arianna/kb"
    )
    chat_services[session_id] = chat_service
    redis_client.set(session_id, json.dumps({}))  # json.dumps serializza i dati in una stringa JSON
    return session_id

def session_exist(session_id):
    return redis_client.exists(session_id)

@chatbot_bp.route("/start_session", methods=["POST"])
def start_session():
    session_id = create_session()
    return jsonify({"session_id": session_id}), 200

@chatbot_bp.route("/ingest_kb", methods=["POST"])
def ingest_kb():
    try:
        data = request.json
        chat_service = ChatService(
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            kb_dir="/Users/luigidaddario/Downloads/kb_zoometa_arianna/kb",
            product_kb_dir="/Users/luigidaddario/Downloads/kb_zoometa_arianna/prodotti"
        )

        if not chat_service.pinecone_manager.index_exists("main-index") or not chat_service.pinecone_manager.index_exists("product-index"):
            chat_service.init_pinecone("main-index", "product-index")

        chat_service.ingest_knowledge_bases()

        return jsonify({"message": "Knowledge base ingested successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@chatbot_bp.route("/delete_indexes", methods=["POST"])
def delete_indexes():
    """Delete existing Pinecone indexes."""
    try:
        chat_service = ChatService(
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            kb_dir="/Users/luigidaddario/Downloads/kb_zoometa_arianna/kb",
            product_kb_dir="/Users/luigidaddario/Downloads/kb_zoometa_arianna/prodotti"
        )

        chat_service.delete_indices("main-index", "product-index")

        return jsonify({"message": "Indexes deleted successfully!"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@chatbot_bp.route("/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id")
    user_input = data.get("input")
    response_format = data.get("response_format", "html")

    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    if not user_input:
        return jsonify({"error": "input is required"}), 400

    if not session_exist(session_id):
        return jsonify({"error": "Invalid session_id"}), 400

    chat_service = chat_services.get(session_id)
    if not chat_service:
        return jsonify({"error": "Chat service not found for session_id"}), 400

    response = chat_service.handle_user_query(user_input, response_format)

    # Salva la conversazione nel database
    conversation = Conversation(session_id=session_id, user_input=user_input, bot_response=response)
    db.session.add(conversation)
    db.session.commit()

    # Notifica in tempo reale
    from sockets.notifications import notify_new_conversation
    notify_new_conversation(conversation)

    return jsonify({"response": response}), 200

@chatbot_bp.route("/rate_chat", methods=["POST"])
def rate_chat():
    try:
        data = request.json
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400
        
        if not session_exist(session_id):
            return jsonify({"error": "Invalid session_id"}), 400
        
        # Salva la recensione nel database
        review = Reviews(session_id=session_id, rating=data.get("rating"), review_text=data.get("comment"))
        db.session.add(review)
        db.session.commit()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"message": "recensione inviata correttamente"}), 200

@chatbot_bp.route("/reset_session", methods=["POST"])
def reset_session():
    try:
        data = request.json
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400

        if not session_exist(session_id):
            return jsonify({"error": "Invalid session_id"}), 400

        chat_service = ChatService(
            pinecone_api_key=os.getenv("PINECONE_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            kb_dir="/Users/luigidaddario/Downloads/auxilium_files_test",
            product_kb_dir="/Users/luigidaddario/Downloads/auxilium_products"
        )
        chat_services[session_id] = chat_service
        chat_service.reset_session()
        redis_client.delete(session_id)

        new_session_id = create_session()

        return jsonify({"message": "Sessione ripristinata con successo!", "new_session_id": new_session_id}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@chatbot_bp.route("/end_session", methods=["POST"])
def end_session():
    try:
        data = request.json
        session_id = data.get("session_id")
        if not session_id:
            return jsonify({"error": "session_id is required"}), 400

        if redis_client.delete(session_id):
            return jsonify({"message": "Sessione terminata con successo!"}), 200
        else:
            return jsonify({"error": "Invalid session_id"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@chatbot_bp.route("/chat_audio", methods=["POST"])
def chat_audio():
    data = request.form
    session_id = data.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id is required"}), 400
    if 'audio' not in request.files:
        return jsonify({"error": "audio file is required"}), 400
    if not session_exist(session_id):
        return jsonify({"error": "Invalid session_id"}), 400

    audio_file = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, 'rb') as f:
        transcript = openai.Audio.transcribe("whisper-1", f)["text"]

    os.remove(tmp_path)

    chat_service = chat_services.get(session_id)
    if not chat_service:
        return jsonify({"error": "Chat service not found for session_id"}), 400

    response_text = chat_service.handle_user_query(transcript, "text")

    conversation = Conversation(session_id=session_id, user_input=transcript, bot_response=response_text)
    db.session.add(conversation)
    db.session.commit()

    from sockets.notifications import notify_new_conversation
    notify_new_conversation(conversation)

    return jsonify({"response": response_text, "transcript": transcript}), 200


@chatbot_bp.route("/text_to_speech", methods=["POST"])
def text_to_speech():
    data = request.json
    text = data.get("text") if data else None
    if not text:
        return jsonify({"error": "text is required"}), 400

    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp:
        tts.save(tmp.name)
        tmp_path = tmp.name

    return_data = send_file(tmp_path, mimetype='audio/mpeg', as_attachment=True, download_name='response.mp3')
    os.remove(tmp_path)
    return return_data
