# config.py
import os
import openai
from dotenv import load_dotenv

load_dotenv()

class Config:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    pinecone_api_key = os.getenv("PINECONE_API_KEY") 
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URI')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SOCKETIO_MESSAGE_QUEUE = os.getenv('SOCKETIO_MESSAGE_QUEUE')  #Redis per SocketIO
    SECRET_KEY = '9UL9jRtREDV1-UuJFtR95hRHnns6trlaXAJgcB5ad5Q'

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True