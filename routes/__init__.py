# routes/__init__.py
from flask import Blueprint

auth_bp = Blueprint('auth', __name__)
chatbot_bp = Blueprint('chatbot', __name__)
dashboard_bp = Blueprint('dashboard', __name__)

from .auth import *
from .chatbot import *
from .dashboard import *