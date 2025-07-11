from flask import Blueprint, jsonify, request
from .utils import *

# Create a blueprint object
api = Blueprint('api', __name__)


@api.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Image Captioning API is live!'})


@api.route('/predict', methods=['POST'])
def predict_caption():
    file = request.files['image']
    path = save_uploaded_image(file)
    # caption = generate_caption(path)
    return jsonify({'message': 'Image uploaded successfully', 'path': path})
    # return jsonify({'caption': caption})


@api.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'pong'}), 200
