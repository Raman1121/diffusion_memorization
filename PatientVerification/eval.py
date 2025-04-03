import os
import csv
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dataset import SiameseNetworkDataset
from model import SiameseNetwork

from sklearn import metrics
import Utils


# Load the trained CKPT
model = SiameseNetwork(network="ResNet-50", in_channels=3, n_features=128).to('cuda')

# Loading ckpt
# CKPT_PATH = "trained_models/best_network.pth"
CKPT_PATH = "/raid/s2198939/diffusion_memorization/PatientVerification/checkpoints/ResNet-50_epoch1_data_handling_RPN.pth"
model.load_state_dict(torch.load(CKPT_PATH))
model.eval()
print("Model loaded successfully!")

# Load the data
CSV_DIR = "/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/Prepared_CSV2"
IMG_DIR = "/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
test_csv = os.path.join(CSV_DIR, "FINAL_TEST.xlsx")

df_test = pd.read_excel(test_csv)
df_test["path"] = df_test["path"].apply(lambda x: os.path.join(IMG_DIR, x))

image_data = list(zip(df_test['path'], df_test['subject_id']))

# Define any transformations (e.g., resizing, normalization).
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Create the dataset and dataloader.
dataset = SiameseNetworkDataset(image_data, transform=transform, data_handling='RPN')
dataloader = DataLoader(dataset, shuffle=True, batch_size=2)

# Evaluate the model on the test set.
def test(model, dataloader):
    y_true = None
    y_pred = None

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            inputs1, inputs2, labels = batch
            inputs1 = inputs1.to('cuda')
            inputs2 = inputs2.to('cuda')
            labels = labels.to('cuda')

            if y_true is None:
                y_true = labels
            else:
                y_true = torch.cat((y_true, labels), 0)

            outputs = model(inputs1, inputs2)
            outputs = torch.sigmoid(outputs)
            score = outputs.cpu()
            
            if y_pred is None:
                y_pred = outputs.cpu()
            else:
                y_pred = torch.cat((y_pred, outputs.cpu()), 0)

    y_pred = y_pred.squeeze()
    return y_true, y_pred, score

y_true, y_pred, score = test(model, dataloader)
y_true, y_pred, score = [y_true.cpu().numpy(), y_pred.cpu().numpy(), score.cpu().numpy()]



