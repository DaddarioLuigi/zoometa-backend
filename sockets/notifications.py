# sockets/notifications.py
from flask_socketio import SocketIO, emit
from models import Conversation

socketio = SocketIO()

@socketio.on('connect')
def handle_connect():
    emit('message', {'data': 'Connessione stabilita'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

def notify_new_conversation(conversation):
    conversation_data = {
        "id": conversation.id,
        "session_id": conversation.session_id,
        "user_input": conversation.user_input,
        "bot_response": conversation.bot_response,
        "timestamp": conversation.timestamp.isoformat()
    }
    socketio.emit('new_conversation', conversation_data)