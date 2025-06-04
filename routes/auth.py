# routes/auth.py
from flask import Blueprint, request, jsonify
from models import db, User
from flask_jwt_extended import create_access_token
from marshmallow import Schema, fields, validate, ValidationError

auth_bp = Blueprint('auth', __name__)

class RegistrationSchema(Schema):
    username = fields.Str(required=True, validate=validate.Length(min=3, max=150))
    password = fields.Str(required=True, validate=validate.Length(min=6))

@auth_bp.route('/register', methods=['POST'])
def register():
    schema = RegistrationSchema()
    try:
        data = schema.load(request.json)
    except ValidationError as err:
        return jsonify(err.messages), 400

    username = data['username']
    password = data['password']

    if User.query.filter_by(username=username).first():
        return jsonify({"error": "Username già esistente"}), 409

    new_user = User(username=username, password=password)
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "Utente registrato con successo"}), 201

# routes/auth.py (aggiungi dopo l'endpoint di registrazione)
class LoginSchema(Schema):
    username = fields.Str(required=True)
    password = fields.Str(required=True)

@auth_bp.route('/login', methods=['POST'])
def login():
    schema = LoginSchema()
    try:
        data = schema.load(request.json)
    except ValidationError as err:
        return jsonify(err.messages), 400

    username = data['username']
    password = data['password']

    user = User.query.filter_by(username=username).first()
    if user and user.check_password(password):
        access_token = create_access_token(identity={'username': user.username, 'role': user.role})
        return jsonify(access_token=access_token), 200
    else:
        return jsonify({"error": "Credenziali non valide"}), 401