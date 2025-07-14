import uuid
from gtts import gTTS
from .config import *


def save_uploaded_image(file):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    ext = file.filename.split('.')[-1]
    filename = f"{uuid.uuid4()}.{ext}"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    return filepath


def caption_to_audio(caption, audio_filename="caption.mp3"):
    tts = gTTS(caption, lang='en')
    full_path = os.path.join(AUDIO_FOLDER, audio_filename)
    tts.save(full_path)
    return full_path
