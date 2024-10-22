# IMPORTS----------------------------------------------------------------------------
# STANDARD
# import sys
import os
import torch
from torch import nn
# import numpy as np
# import matplotlib.pyplot as plt
import shutil
# from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import pandas as pd
import wandb

# CUSTOM
from network import *
from utils import *
from dataloader import *
# import pdb
# import utils

# Load the parameters
from loadParam import *

if os.path.exists(JOB_FOLDER):
    shutil.rmtree(JOB_FOLDER)
    print(f"deleted previous job folder from {JOB_FOLDER}")
os.mkdir(JOB_FOLDER)
os.mkdir(TRAINED_MDL_PATH)

# DATASET ---------------------------------------------------------------------------
datatype = torch.float32
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

# Define the dataset size
dataset = WindowDataset(DS_PATH)

# Split the dataset into train and validation
dataset_size = len(dataset)

train_size = int(0.9 * dataset_size)
test_size = dataset_size - train_size
trainset, valset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
trainLoader = torch.utils.data.DataLoader(trainset, BATCH_SIZE, False, num_workers=NUM_WORKERS)
valLoader = torch.utils.data.DataLoader(valset, BATCH_SIZE, False, num_workers=NUM_WORKERS)

# Network and optimzer --------------------------------------------------------------
model = Network(3, 1)  
model = model.to(device)

# predicted_mask =torch.sigmoid(model)

# LOSS FUNCTION AND OPTIMIZER
optimizer = optim.Adam(model.parameters(), lr=LR)   # we can modify lr start wih 1e-4 or 1e-3

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 0.5 ** (epoch // 10))

def shouldLog(batchcount=None):
    if batchcount==None:
        return LOG_WANDB=='true'
    else:
        return batchcount%LOG_BATCH_INTERVAL == 0 and LOG_WANDB=='true'

# INIT LOGGER
wandb.init(
    project=MODEL_NAME,
    name=str(JOB_ID),
    
    # track hyperparameters and run metadata
    config={
    "JOB_ID":JOB_ID,
    "learning_rate": LR,
    "batchsize": BATCH_SIZE,
    "dataset": DS_PATH,
    }
)

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

def calculate_accuracy(output, target):
    output = torch.sigmoid(output)
    predicted = (output > 0.5).float()
    correct = (predicted == target).float().sum()
    return correct / target.numel()

#  TRAIN ----------------------------------------------------------------------------
def train(dataloader, model, loss_fn, optimizer, epochstep):
    
    # dp('train started')
    model.train()
    total_loss = 0
    total_acc = 0
    # epochloss = 0

    for batchcount, (rgb, label) in enumerate(dataloader):
        dp(' batch', batchcount)
        
        rgb = rgb.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()
        
        pred = model(rgb)

        # Change after error
        # pred_resized = torch.nn.functional.interpolate(pred, size=label.shape[2:], mode='bilinear', align_corners=False)

        # Change after error
        loss = loss_fn(pred, label.float())
        # loss = loss_fn(pred_resized, label.float())        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += calculate_accuracy(pred, label).item()

        # epochloss += loss.item()

        wandb.log({
            "epochstep": epochstep,
            "batch/loss/train": loss.item(),
                })
            
        if batchcount == 0: # only for the first batch every epoch
            wandb_images = []
            for (pred_single, label_single, rgb_single) in zip(pred, label, rgb):
                combined_image_np = CombineImages(pred_single, label_single, rgb_single)

                # Create wandb.Image object and append to the list
                wandb_images.append(wandb.Image(combined_image_np))

            wandb.log(
            {
                "images/train": wandb_images,
            })
                    
    if shouldLog():
        wandb.log({
            "epoch/loss/train": total_loss,
                    })
        

    return total_loss / len(dataloader), total_acc / len(dataloader)
    
# Define the val function
def val(dataloader, model, loss_fn, epochstep):
    model.eval()
    
    total_loss = 0
    total_acc = 0

    # epochloss = 0
    with torch.no_grad():
        for batchcount, (rgb, label) in enumerate(dataloader):
            dp(' batch', batchcount)
            
            rgb = rgb.to(device)
            label = label.to(device)
            
            pred = model(rgb)

            # Resize prediction to match the target size (label)
            # pred_resized = torch.nn.functional.interpolate(pred, size=label.shape[2:], mode='bilinear', align_corners=False)

            # Compute the loss with resized prediction
            loss = loss_fn(pred, label.float())
            total_loss += loss.item()
            total_acc += calculate_accuracy(pred, label).item()
            # loss = loss_fn(pred_resized, label.float())      

            # epochloss += loss.item()
        
            wandb.log({
                "batch/loss/": loss.item(),
                    })
            
            if batchcount == 0: # only for the first batch every epoch
                wandb_images = []
                for (pred_single, label_single, rgb_single) in zip(pred, label, rgb):
                    combined_image_np = CombineImages(pred_single, label_single, rgb_single)
                    wandb_images.append(wandb.Image(combined_image_np))

                wandb.log(
                {
                    "images/val": wandb_images,
                })
            
    wandb.log({
        "epoch/loss/val": total_loss,
                })
    
    return total_loss / len(dataloader), total_acc / len(dataloader)

# STORE ORIGINAL PARAMTERS
trainedMdlPath = TRAINED_MDL_PATH + f"test.pth"
torch.save(model.state_dict(), trainedMdlPath)

# SCRIPT ---------------------------------------------------------------------------------
epochs = 100   # can be changed

lossFn = loss_func = nn.BCEWithLogitsLoss()

# lossFn = FocalLoss(alpha=[0.75, 0.25], gamma=2, logits=True, reduce=True)

for eIndex in range(epochs):
    dp(f"Epoch {eIndex+1}\n")

    train_loss, train_accuracy = train(trainLoader, model, lossFn, optimizer, eIndex)
    val_loss, val_accuracy = val(valLoader, model, lossFn, eIndex)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    trainedMdlPath = TRAINED_MDL_PATH + f"{eIndex}.pth"
    torch.save(model.state_dict(), trainedMdlPath)

    scheduler.step()


# Export the metrics to a CSV file
metrics_data = {
    'Train Loss': train_losses,
    'Validation Loss': val_losses,
    'Train Accuracy': train_accuracies,
    'Validation Accuracy': val_accuracies
}
df = pd.DataFrame(metrics_data)
df.to_csv('./resnet18_unet/training_metrics.csv', index=False)

# Evaluate on the test set
# test_loss, test_acc = val(valLoader, model, lossFn)
# print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")