# utils/helpers.py
from functools import wraps
from flask_jwt_extended import verify_jwt_in_request, get_jwt_identity
from flask import jsonify

def role_required(required_role):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            verify_jwt_in_request()
            identity = get_jwt_identity()
            user_role = identity.get('role', None)
            if user_role != required_role:
                return jsonify({"error": "Accesso non autorizzato"}), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator