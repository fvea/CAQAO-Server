from db import db
from datetime import datetime

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(80), unique=True, nullable=False)
    last_name = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(80), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(128), nullable=False)

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image = db.Column(db.LargeBinary, nullable=False)
    filename = db.Column(db.Text, nullable=False)
    mimetype = db.Column(db.Text, nullable=False)
    beanGrade = db.Column(db.Text, nullable=False)
    veryDarkBrown = db.Column(db.Integer, default=0)
    brown = db.Column(db.Integer, default=0)
    partlyPurple = db.Column(db.Integer, default=0)
    totalPurple = db.Column(db.Integer, default=0)
    g1 = db.Column(db.Integer, default=0)
    g2 = db.Column(db.Integer, default=0)
    g3 = db.Column(db.Integer, default=0)
    g4 = db.Column(db.Integer, default=0)
    mouldy = db.Column(db.Integer, default=0)
    insectInfested = db.Column(db.Integer, default=0)
    slaty = db.Column(db.Integer, default=0)
    germinated = db.Column(db.Integer, default=0)
    date = db.Column(db.DateTime, default=datetime.utcnow)

class TempDetection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image = db.Column(db.LargeBinary, nullable=False)
    filename = db.Column(db.Text, nullable=False)
    mimetype = db.Column(db.Text, nullable=False)
    beanGrade = db.Column(db.Text, nullable=False)
    veryDarkBrown = db.Column(db.Integer, default=0)
    brown = db.Column(db.Integer, default=0)
    partlyPurple = db.Column(db.Integer, default=0)
    totalPurple = db.Column(db.Integer, default=0)
    g1 = db.Column(db.Integer, default=0)
    g2 = db.Column(db.Integer, default=0)
    g3 = db.Column(db.Integer, default=0)
    g4 = db.Column(db.Integer, default=0)
    mouldy = db.Column(db.Integer, default=0)
    insectInfested = db.Column(db.Integer, default=0)
    slaty = db.Column(db.Integer, default=0)
    germinated = db.Column(db.Integer, default=0)
    date = db.Column(db.DateTime, default=datetime.utcnow)