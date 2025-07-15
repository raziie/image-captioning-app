from app import app


from data.dataloader import get_dataloaders
from config.app_config import *
import pickle

train_loader, val_loader, test_loader, vocab = get_dataloaders()
with open(VOCAB_PATH, 'wb') as f:
    pickle.dump(vocab, f)

if __name__ == '__main__':
    app.run(debug=True)
