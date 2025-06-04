# app.py
from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
from flask_migrate import Migrate
from flask_socketio import SocketIO
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

from config import DevelopmentConfig
from models import db
from routes import auth_bp, chatbot_bp, dashboard_bp
from sockets.notifications import socketio as socketio_instance

import os 

def create_app():
    app = Flask(__name__)
    app.config.from_object(DevelopmentConfig)

    # Inizializza estensioni
    CORS(app)
    db.init_app(app)
    Migrate(app, db)
    JWTManager(app)
    socketio_instance.init_app(app, cors_allowed_origins="*")

    # Registra blueprint
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(chatbot_bp, url_prefix='/chatbot')
    app.register_blueprint(dashboard_bp, url_prefix='/dashboard')

    return app

app = create_app() 

if __name__ == "__main__":
    socketio_instance.run(app, debug=True)