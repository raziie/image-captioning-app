import uuid
from gtts import gTTS
from config.app_config import *


def save_uploaded_image(file):
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    ext = file.filename.split('.')[-1]
    base_filename = str(uuid.uuid4())
    filename = f"{base_filename}.{ext}"
    filepath = os.path.join(IMAGE_FOLDER, filename)
    file.save(filepath)
    return filepath, base_filename


def save_caption_audio(caption, base_filename):
    os.makedirs(AUDIO_FOLDER, exist_ok=True)
    filename = f"{base_filename}.mp3"
    filepath = os.path.join(AUDIO_FOLDER, filename)
    tts = gTTS(caption, lang='en')
    tts.save(filepath)
    return filepath
