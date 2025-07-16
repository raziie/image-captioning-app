import uuid
from gtts import gTTS
from config.app_config import *


def save_uploaded_image(file):
    ext = file.filename.split('.')[-1]
    base_filename = str(uuid.uuid4())
    filename = f"{base_filename}.{ext}"
    relative_path = os.path.join('static', 'uploads', 'image', filename)
    filepath = os.path.join(BASE_APP_DIR, relative_path)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    file.save(filepath)
    return relative_path, base_filename


def save_caption_audio(caption, base_filename):
    filename = f"{base_filename}.mp3"
    relative_path = os.path.join('static', 'uploads', 'audio', filename)
    filepath = os.path.join(BASE_APP_DIR, relative_path)
    tts = gTTS(caption, lang='en')
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    tts.save(filepath)
    return relative_path
