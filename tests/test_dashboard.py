import os
import sys
import types
import pytest
from flask_jwt_extended import create_access_token

# Stub out heavy optional dependencies before importing app
services_pkg = types.ModuleType('services')
for mod_name in ['ChatService', 'PineconeManager', 'DocumentIngestion', 'ChatbotAgent', 'CustomRecommendationTool']:
    sub = types.ModuleType(f'services.{mod_name}')
    cls = type(mod_name, (), {})
    sub.__dict__[mod_name] = cls
    setattr(services_pkg, mod_name, cls)
    sys.modules[f'services.{mod_name}'] = sub
sys.modules['services'] = services_pkg

# Stub external libraries used by service modules
for name in [
    'llama_index',
    'llama_index.core',
    'llama_index.core.settings',
    'llama_index.core.indices',
    'llama_index.core.indices.vector_store',
    'llama_index.core.tools',
    'llama_index.embeddings',
    'llama_index.embeddings.openai',
    'llama_index.llms',
    'llama_index.llms.openai',
    'llama_index.vector_stores',
    'llama_index.vector_stores.pinecone',
]:
    module = types.ModuleType(name)
    sys.modules[name] = module


# Ensure the application uses an in-memory database during import
# Force the app to use an in-memory SQLite DB for testing even if a
# DATABASE_URI is already defined in the environment.
os.environ['DATABASE_URI'] = 'sqlite:///:memory:'

pinecone = types.ModuleType('pinecone')
pinecone.Pinecone = object
pinecone.ServerlessSpec = object
sys.modules['pinecone'] = pinecone

from app import create_app
from models import db, Conversation

@pytest.fixture
def app_instance():
    app = create_app()
    app.config.update(
        TESTING=True,
        SQLALCHEMY_DATABASE_URI="sqlite:///:memory:",
        JWT_SECRET_KEY="testing-secret"
    )
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def client(app_instance):
    return app_instance.test_client()


def test_conversations_authorized(client, app_instance):
    with app_instance.app_context():
        conv = Conversation(session_id="s1", user_input="hi", bot_response="hello")
        db.session.add(conv)
        db.session.commit()
        token = create_access_token(identity="tester")
    headers = {"Authorization": f"Bearer {token}"}
    response = client.get("/dashboard/conversations", headers=headers)
    assert response.status_code == 200
