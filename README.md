
# 🖼️ Image Caption Generator

Automatically generate human-like captions for images using deep learning. This project combines computer vision and natural language processing to describe images in natural English. Bonus: it can speak the caption out loud!

---

## 🚀 Features

- 📷 Upload an image and get a caption generated automatically.
- 🔈 Convert the caption into audio using text-to-speech (gTTS).
- 🤖 Uses a custom-trained Encoder-Decoder architecture with attention mechanism.
- 🧠 Trained on the **Flickr8k** dataset with **Karpathy splits**.
- 🗣️ Tokenized with spaCy and saved in a custom `Vocabulary` class.

---

## 🧱 Project Structure

```
image-captioning/
│
├── app/
│   ├── static/
│        └── uploads/        # Stores uploaded images & audio files
│   ├── inference.py         # Inference logic and model loading
│   ├── routes.py            # Web routes for Flask app
│   ├── utils.py             # Helper functions like save_uploaded_image(), save_caption_audio()
│
├── captioning/
│   ├── generator.py         # Generates caption using beam search
│   ├── visualize.py         # Visualize attention mechanism
│
├── checkpoints/
│   ├── checkpoint_best.pth  # Best model checkpoint
│   ├── vocab.pkl            # Pickled vocabulary
│
├── config/
│   ├── app_config.py       # App-specific parameters
│   ├── base_config.py      # Base config shared across models
│   ├── data_config.py      # Data-specific parameters
│   ├── train_config.py     # Train-specific parameters
│
├── data/
│   ├── input/               # Input dataset files
│   ├── collate.py           # Collate class
│   ├── data_analysis.py     # Functions for data analysis
│   ├── dataloader.py        # DataLoader setup
│   ├── vocab.py             # Vocabulary class
│   ├── dataset.py           # Custom Image Captioning Dataset 
│
├── engine/
│   ├── setup.py              # Functions to get optimizers, schedulers, and criterion
│   ├── train.py              # Training loop
│   ├── validate.py           # Validation logic
│   ├── evaluate.py           # Evaluation/metrics
│
├── frontend/
│   ├── node_modules/         # Dependencies installed by npm (auto-generated)
│   ├── src/                  # Source code for the React frontend
│   │ ├── App.css             # Styles specific to the App component 
│   │ ├── App.jsx             # Main App component
│   │ ├── main.jsx            # Entry point that renders <App /> into the DOM
│   │   
│   ├── eslint.config.js      # ESLint configuration for code linting and formatting
│   ├── index.html            # HTML template used by Vite to inject the app
│   ├── package.json          # Project metadata and dependencies
│   ├── package-lock.json     # Exact dependency versions locked for consistent installs
│   ├── vite.config.js        # Vite configuration for the build tool and dev server
│   
├── notebooks/
│   ├── image-caption-generation.ipynb
│
├── models/
│   ├── attention.py         # Attention class
│   ├── encoder.py           # Encoder CNN (e.g., ResNet)
│   ├── decoder.py           # Decoder LSTM with attention
│  
├── run.py                   # Main runner (Flask app)      
├── train_run.py             # Main training script for captioning model (load, train, validate, evaluate) 
├── requirements.txt         # Python dependencies required to run the backend
└── README.md                # You’re reading it 😄
```

---

## 🧪 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## 📂 Dataset Setup

1. Download the **Flickr8k** image dataset and **Karpathy splits**:
   - Place images in: `data/flickr8k/images/`
   - Place JSON in: `data/flickr8k/dataset_flickr8k.json`

2. First run will tokenize captions and create the vocabulary.

---

## 🏁 Running the App

### 💻 Train the model
```bash
python run_train.py
```

### 🖼️ Run inference (Flask)
```bash
python run.py
```

### 🚀 Run UI (REACT)
```bash
cd ./frontend
npm run dev
```

Then upload an image to get its caption and audio version.

---

## 🔧 Configuration

Edit configs in the `configs/` directory to tweak model settings, learning rates, beam size, etc.

---

## 📈 Model Architecture

- **Encoder**: Pretrained ResNet-101 (fine-tunable)
- **Decoder**: LSTM with attention
- **Tokenizer**: spaCy-based with custom `Vocabulary` class

---

## 📣 Credits

- Dataset: [Flickr8k](https://github.com/jbrownlee/Datasets)
- Tokenizer: [spaCy](https://spacy.io/)
- TTS: [gTTS](https://pypi.org/project/gTTS/)
- Inspired by the [Show, Attend and Tell](https://arxiv.org/abs/1502.03044) paper

---

## 🧠 Future Improvements

- Replace GRU with Transformer decoder
- Support for multilingual captions

---

## 🛡️ License

MIT License. Do cool stuff, just give credit. 🙌
