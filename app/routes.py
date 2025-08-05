from flask import Blueprint, jsonify, request, send_file
from .inference import *

# Create a blueprint object
api = Blueprint('api', __name__)


@api.route('/', methods=['GET'])
def index():
    return jsonify({'message': 'Image Captioning API is live!'})


@api.route('/predict', methods=['POST'])
def predict_caption():
    file = request.files['image']
    path, base_filename = save_uploaded_image(file)
    caption, attention_map_plot = generate_caption(path, visualize=True, base_filename=base_filename)
    audio_path = save_caption_audio(caption, base_filename)
    return jsonify({
        'caption': caption,
        'audio': audio_path.replace("\\", "/"),
        'attention_map_plot': attention_map_plot.replace("\\", "/")
    })


# @api.route('/static/uploads/audio/<filename>')
# def download_audio(filename):
#     return send_from_directory(
#         directory=AUDIO_DIR,
#         path=filename,
#         as_attachment=True
#     )


@api.route('/ping', methods=['GET'])
def ping():
    return jsonify({'message': 'pong'}), 200
