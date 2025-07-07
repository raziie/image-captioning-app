from flask import Flask
from .routes import api
from flask_cors import CORS

app = Flask(__name__)
app.register_blueprint(api)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
# Allow both GET and POST from React dev server
CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})