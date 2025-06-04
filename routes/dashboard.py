# routes/dashboard.py
from flask import Blueprint, jsonify, request, send_file
from flask_jwt_extended import jwt_required
from models import db, Conversation
from utils.helpers import role_required
import io
import csv

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('/admin_only', methods=['GET'])
@jwt_required()
@role_required('admin')
def admin_only_route():
    return jsonify({"message": "Solo admin possono vedere questo"}), 200


@dashboard_bp.route("/conversations", methods=["GET"])
@jwt_required()
def get_conversations():
    # retrieve conversations ordered by newest first
    conversations = Conversation.query.order_by(Conversation.timestamp.desc()).all()
    conversations_data = [
        {
            "id": conv.id,
            "session_id": conv.session_id,
            "user_input": conv.user_input,
            "bot_response": conv.bot_response,
            "timestamp": conv.timestamp.isoformat()
        } for conv in conversations
    ]
    return jsonify({"conversations": conversations_data}), 200

@dashboard_bp.route("/conversations/download", methods=["GET"])
@jwt_required()
def download_conversations():
    conversations = Conversation.query.all()
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['ID', 'Session ID', 'User Input', 'Bot Response', 'Timestamp'])
    for conv in conversations:
        cw.writerow([conv.id, conv.session_id, conv.user_input, conv.bot_response, conv.timestamp.isoformat()])
    
    output = io.BytesIO()
    output.write(si.getvalue().encode('utf-8'))
    output.seek(0)
    
    return send_file(output, mimetype='text/csv', as_attachment=True, download_name='conversations.csv')

@dashboard_bp.route("/statistics", methods=["GET"])
@jwt_required()
def get_statistics():
    total_users = db.session.query(db.func.count(Conversation.id)).scalar()
    total_conversations = db.session.query(db.func.count(Conversation.id)).scalar()
    successes = db.session.query(db.func.count(Conversation.id)).filter(Conversation.bot_response.isnot(None)).scalar()
    errors = total_conversations - successes

    messages_per_day = db.session.query(
        db.func.date(Conversation.timestamp).label('day'),
        db.func.count(Conversation.id).label('count')
    ).group_by('day').all()

    messages_per_day_data = [{"day": day, "count": count} for day, count in messages_per_day]

    stats = {
        "totalUsers": total_users,
        "totalConversations": total_conversations,
        "successes": successes,
        "errors": errors,
        "messagesPerDay": messages_per_day_data
    }

    return jsonify(stats), 200

@dashboard_bp.route("/summary", methods=["GET"])
@jwt_required()
def get_summary():
    total_users = db.session.query(db.func.count(db.func.distinct(Conversation.session_id))).scalar()
    total_messages = db.session.query(db.func.count(Conversation.id)).scalar()
    successes = db.session.query(db.func.count(Conversation.id)).filter(Conversation.bot_response.isnot(None)).scalar()
    errors = total_messages - successes

    summary = {
        "totalUsers": total_users,
        "totalMessages": total_messages,
        "successes": successes,
        "errors": errors
    }

    return jsonify(summary), 200

@dashboard_bp.route("/settings", methods=["POST"])
@jwt_required()
def update_settings():
    # Implementa la logica per aggiornare le impostazioni
    data = request.json
    # Esempio: aggiornare il nome del chatbot
    chatbot_name = data.get("chatbotName")
    if chatbot_name:
        # Aggiorna le impostazioni in database o in file di configurazione
        # Questo è un esempio e potrebbe variare a seconda delle tue necessità
        return jsonify({"message": "Impostazioni aggiornate con successo"}), 200
    else:
        return jsonify({"error": "Parametri mancanti"}), 400
    

@dashboard_bp.route("/conversations/details/<session_id>", methods=["GET"])
@jwt_required()
def get_conversation_details(session_id):
    # Cerca tutte le conversazioni con il session_id specificato
    conversations = Conversation.query.filter_by(session_id=session_id).all()

    if not conversations:
        return jsonify({"error": "Nessuna conversazione trovata con il session_id specificato"}), 404

    # Prepara i dati per la risposta
    conversations_data = [
        {
            "id": conv.id,
            "session_id": conv.session_id,
            "user_input": conv.user_input,
            "bot_response": conv.bot_response,
            "timestamp": conv.timestamp.isoformat()
        } for conv in conversations
    ]

    return jsonify({"messages": conversations_data}), 200