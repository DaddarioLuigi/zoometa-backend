from . import db
from datetime import datetime,timezone


class Reviews(db.Model):
    __tablename__ = 'reviews'

    review_id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(db.String(36), index=True, nullable=False)
    rating = db.Column(db.Integer, nullable=True)
    review_text = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=db.func.now())

    def __init__(self, session_id, rating, review_text):
        self.session_id = session_id
        self.rating = rating
        self.review_text = review_text