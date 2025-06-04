# models/__init__.py
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

from .user import User
from .conversation import Conversation
from .reviews import Reviews