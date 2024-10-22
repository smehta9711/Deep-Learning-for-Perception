import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
# from dataaug import *
from loadParam import *
import pdb


class WindowDataset(Dataset):
    
    def __init__(self, ds_path,img_transform=None, mask_transform = None, image_size = (256,256),default_size = (640,360)):
        
        # self.img_dir= os.path.join(ds_path,"original_images")
        # self.mask_dir = os.path.join(ds_path,"segmented_mask")
        self.img_dir= os.path.join(ds_path,"og")
        self.mask_dir = os.path.join(ds_path,"mask")
        
        self.img_transform=img_transform
        self.mask_transform = mask_transform
        self.image_size=image_size
        self.default_size= default_size
        self.img_files = os.listdir(self.img_dir)
        self.mask_files = os.listdir(self.mask_dir)
        
    def __len__(self):
        # Set the dataset size here
        return len(self.img_files)

    def __getitem__(self, idx):
        # idx is from 0 to N-1
        
        img_path = os.path.join(self.img_dir,self.img_files[idx])
        mask_path = os.path.join(self.mask_dir,self.mask_files[idx])
        
        # Open the RGB image and ground truth label
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Assuming mask is grayscale

        # convert them to tensors
        
        #if self.img_transform==None:
        image = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to match the input size for U-Net
        transforms.ToTensor(),          # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
        ])(image)

        # Apply mask transformation
        #if self.mask_transform==None:
        mask = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to match the input size for U-Net
        transforms.ToTensor()           # Convert mask to PyTorch tensor
        ])(mask)
            

        # apply any transform (blur, noise...)
        
        return image, mask

        
# Define image transformations
img_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match the input size for U-Net
    transforms.ToTensor(),          # Convert image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Define mask transformations
mask_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to match the input size for U-Net
    transforms.ToTensor()           # Convert mask to PyTorch tensor
])       
        
        

# verify the dataloader
if __name__ == "__main__":
    dataset = WindowDataset(ds_path= DS_PATH)
    dataloader = DataLoader(dataset, batch_size=8)

     # rgb, label = dataset[0]
