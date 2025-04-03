import os
import csv
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from dataset import SiameseNetworkDataset
from model import SiameseNetwork

def parse_args():
    parser = argparse.ArgumentParser(description='Patient Verification')
    parser.add_argument('--data_handling', type=str, default='RPN', help='Data handling mode: FTS or RPN')
    parser.add_argument('--fixed_size', type=int, default=None, help='Fixed number of pairs to generate per epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--network', type=str, default='ResNet-50', help='Network architecture')
    parser.add_argument('--in_channels', type=int, default=3, help='Number of input channels')
    parser.add_argument('--n_features', type=int, default=128, help='Number of output features')
    parser.add_argument('--save_dir', type=str, default='checkpoints/', help='Directory to save model checkpoints')

    parser.add_argument('--image_size', type=int, default=256, help='Size of the input images')
    parser.add_argument('--transform', type=str, default='image_net', help='Type of image transformation')
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the data
    CSV_DIR = "/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0/Prepared_CSV2"
    IMG_DIR = "/raid/s2198939/MIMIC_Dataset/physionet.org/files/mimic-cxr-jpg/2.0.0"
    train_csv = os.path.join(CSV_DIR, "FINAL_TRAIN.xlsx")
    test_csv = os.path.join(CSV_DIR, "FINAL_TEST.xlsx")

    print("Loading train and test CSV files...")
    df_train = pd.read_excel(train_csv)
    df_test = pd.read_excel(test_csv)
    print("Done!")

    # Update the 'path' column to include the full image path.
    df_train["path"] = df_train["path"].apply(lambda x: os.path.join(IMG_DIR, x))
    df_test["path"] = df_test["path"].apply(lambda x: os.path.join(IMG_DIR, x))

    # Create list of (image_path, patient_id) tuples for training and validation.
    image_data_train = list(zip(df_train['path'], df_train['subject_id']))
    image_data_val = list(zip(df_test['path'], df_test['subject_id']))

    # Initialize transformations
    if args.transform == 'image_net':
        transform_train = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        transform_val_test = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ])
        transform_val_test = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor()
        ])

    train_dataset = SiameseNetworkDataset(
        image_data_train,
        data_handling=args.data_handling,
        fixed_size=args.fixed_size,
        transform=transform_train
    )
    val_dataset = SiameseNetworkDataset(
        image_data_val,
        data_handling=args.data_handling,
        fixed_size=args.fixed_size,
        transform=transform_val_test
    )
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize the model
    model = SiameseNetwork(network=args.network, in_channels=args.in_channels, n_features=args.n_features).to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create directory for logs if it doesn't exist.
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    csv_log_path = os.path.join(logs_dir, "loss_log.csv")

    print('Starting training...')
    with open(csv_log_path, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Epoch", "Train Loss", "Validation Loss"])

        # Training loop.
        for epoch in range(args.num_epochs):
            model.train()
            running_loss = 0.0
            for i, (img1, img2, labels) in enumerate(train_dataloader):
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(img1, img2)
                loss = criterion(outputs, labels.float())
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i % 10 == 0:
                    print(f'Epoch [{epoch + 1}/{args.num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')

            avg_train_loss = running_loss / len(train_dataloader)
            print(f"Epoch [{epoch + 1}/{args.num_epochs}] Training Loss: {avg_train_loss:.4f}")

            # Validation loop.
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for img1, img2, labels in val_dataloader:
                    img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                    outputs = model(img1, img2)
                    loss = criterion(outputs, labels.float())
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Epoch [{epoch + 1}/{args.num_epochs}] Validation Loss: {avg_val_loss:.4f}")

            # Write the epoch's losses to the CSV log.
            csv_writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])
            csvfile.flush()  # Ensure data is written to disk

            # Save model checkpoint.
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            checkpoint_path = os.path.join(args.save_dir, f'{args.network}_epoch{epoch + 1}_data_handling_{args.data_handling}.pth')

            if(epoch % 10 == 0 or epoch == args.num_epochs - 1):
                torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    print('Training complete.')
    print(f"Training log saved at {csv_log_path}")

if __name__ == "__main__":
    main()