import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from modules.custom_CheXpert_Dataset import CheXpert
from modules.classifier import Classifier
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()
chexpert_root = os.getenv('PATH_TO_CHEXPERT_ROOT')

# Define important filenames.
train_dinfo_filename = 'train_data_info.csv'
train_images_dirname = 'train_data'

valid_dinfo_filename = 'valid_data_info.csv'
vaild_images_dirname = 'valid_data'

# Hyperparameters
learning_rate = 0.001
momentum = 0.9
num_epochs = 10

im_size = (320, 320) # Input image size
l1_out_chann = 8 # Number of channels in the first convolutional layer
l2_out_chann = 16 # Number of channels in the second convolutional layer
l3_out_chann = 32  # Number of channels in the third convolutional layer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the training dataset
train_dataset = CheXpert(root_dir = chexpert_root,
                    dinfo_filename = train_dinfo_filename,
                    images_dirname = train_images_dirname,
                    resize = True,
                    custom_size = im_size,
                    to_grayscale = True,
                    custom_transforms = transforms.ToTensor())

# Define the test dataset size as the 30% of the train dataset.
test_size = int(0.3 * len(train_dataset))
# Redefine the train dateset size
train_size = len(train_dataset) - test_size

# Split the training dataset into training and test sets
train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])


# Initialize the validation dataset
valid_dataset = CheXpert(root_dir = chexpert_root,
                         dinfo_filename = valid_dinfo_filename,
                         images_dirname = vaild_images_dirname,
                         resize = True,
                         custom_size = im_size,
                         to_grayscale = True,
                         custom_transforms = transforms.ToTensor())

# Initialize the model
net = Classifier(l1_out_chann = l1_out_chann,
                 l2_out_chann = l2_out_chann,
                 l3_out_chann = l3_out_chann,
                 im_size = im_size)

net.to(device)
train_dataset.to(device)
valid_dataset.to(device)
test_dataset.to(device)

# Create data loaders
train_loader = DataLoader(dataset = train_dataset,
                          batch_size = 64,
                          shuffle = True)

valid_loader = DataLoader(dataset = valid_dataset,
                          batch_size = 64,
                          shuffle = False)

test_loader = DataLoader(dataset = test_dataset,
                          batch_size = 64,
                          shuffle = False)

# Define the loss function (CrossEntropyLoss) and optimizer (SGD)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SDG(net.parameters(), lr = learning_rate, momentum = momentum)

# Training loop
for epoch in range(num_epochs):
    net.train()
    total_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    av_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {av_loss:.4f}")

    net.eval()
    correct_preds = 0
    total_preds = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = net(images)
            _, correct_indxs = torch.max(labels, 1)
            _, predictions = torch.max(outputs, 1)
            total_preds += labels.size(0)
            correct_preds += (predictions == correct_indxs).sum().item()
    
    accuracy = 100 * correct_preds / total_preds
    print(f'Validation accuracy: {accuracy:.2f}%')

print("Training finished")

# Evaluate the model on the test dataset
net.eval()

correct_preds = 0
total_preds = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = net(images)
        _, correct_indxs = torch.max(labels, 1)
        _, predictions = torch.max(outputs, 1)
        total_preds += labels.size(0)
        correct_preds += (predictions == correct_indxs).sum().item()
    
accuracy = 100 * correct_preds / total_preds
print(f'Test accuracy: {accuracy:.2f}%')