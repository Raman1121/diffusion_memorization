import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class SiameseNetworkDataset(Dataset):
    def __init__(self, image_data, transform=None, data_handling='RPN', fixed_size=None):
        """
        Args:
            image_data (list of tuples): Each tuple is (image_path, patient_id).
            transform (callable, optional): Transformations to apply on the images.
            data_handling (str): 'FTS' for Fixed Training Sets or 'RPN' for Randomized Negative Pairs.
            fixed_size (int, optional): In RPN mode, sets the fixed number of pairs to generate per epoch.
        """
        self.image_data = image_data
        self.transform = transform
        self.data_handling = data_handling.upper()  # Normalize mode string
        self.fixed_size = fixed_size
        
        # Build a mapping from patient IDs to a list of corresponding image paths.
        self.patient_to_images = {}
        for img_path, patient_id in self.image_data:
            self.patient_to_images.setdefault(patient_id, []).append(img_path)
        self.patient_ids = list(self.patient_to_images.keys())
        
        if self.data_handling == 'FTS':
            # Precompute a fixed list of pairs.
            self.fixed_pairs = []
            for img1_path, patient_id1 in self.image_data:
                # Generate a positive pair if possible.
                if len(self.patient_to_images[patient_id1]) > 1:
                    candidate_paths = self.patient_to_images[patient_id1].copy()
                    candidate_paths.remove(img1_path)
                    pos_img2_path = random.choice(candidate_paths)
                    self.fixed_pairs.append((img1_path, pos_img2_path, 1))
                # Generate a negative pair.
                other_patient_ids = self.patient_ids.copy()
                other_patient_ids.remove(patient_id1)
                neg_patient = random.choice(other_patient_ids)
                neg_img2_path = random.choice(self.patient_to_images[neg_patient])
                self.fixed_pairs.append((img1_path, neg_img2_path, 0))
            
            # Shuffle the fixed pairs so that the order is random.
            random.shuffle(self.fixed_pairs)

    def __len__(self):
        if self.data_handling == 'FTS':
            return len(self.fixed_pairs)
        else:  # RPN mode
            # If a fixed_size is provided, use it; otherwise default to the size of the base image data.
            return self.fixed_size if self.fixed_size is not None else len(self.image_data)

    def __getitem__(self, index):
        if self.data_handling == 'FTS':
            img1_path, img2_path, label = self.fixed_pairs[index]
        else:
            # RPN mode: generate pairs on the fly.
            # Use index modulo len(image_data) to choose a base image.
            base_index = index % len(self.image_data)
            img1_path, patient_id1 = self.image_data[base_index]
            # Decide randomly whether to create a positive pair.
            same_class = random.choice([True, False])
            if same_class and len(self.patient_to_images[patient_id1]) > 1:
                candidate_paths = self.patient_to_images[patient_id1].copy()
                candidate_paths.remove(img1_path)
                img2_path = random.choice(candidate_paths)
                label = 1
            else:
                other_patient_ids = self.patient_ids.copy()
                other_patient_ids.remove(patient_id1)
                neg_patient = random.choice(other_patient_ids)
                img2_path = random.choice(self.patient_to_images[neg_patient])
                label = 0

        # print("Image 1 path:", img1_path)
        # print("Image 2 path:", img2_path)
        
        # Load images and apply any transformations.
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return img1, img2, torch.tensor([label], dtype=torch.float32)