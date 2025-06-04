# models/conversation.py
from . import db
from datetime import datetime,timezone

class Conversation(db.Model):
    __tablename__ = 'conversations'
    
    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), index=True, nullable=False)
    user_input = db.Column(db.Text, nullable=False)
    bot_response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.now())

    def __init__(self, session_id, user_input, bot_response):
        self.session_id = session_id
        self.user_input = user_input
        self.bot_response = bot_response