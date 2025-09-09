import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from dataset import ResumeDataset
from model import ResumeModel

# Load validation dataset
val_dataset = ResumeDataset('data/val/')
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Load model and weights
model = ResumeModel()
model.load_state_dict(torch.load('models/model_weights.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds))
