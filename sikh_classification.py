import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
import shutil

# Set up device for training (GPU if available, otherwise CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Data augmentation and normalization for training and validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Set up data directories
data_dir = 'data/images'

# Create a validation split from training data (80% train, 20% validation)
# We don't have a dedicated validation set, so we'll create one

def create_train_val_split(data_dir, train_ratio=0.8):
    """
    Creates train and validation datasets by splitting the training data.
    """
    # Path to full training data
    train_dir = os.path.join(data_dir, 'train')
    
    # Create train and val directories if they don't exist
    os.makedirs(os.path.join(data_dir, 'train_split'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'val'), exist_ok=True)
    
    # Get classes (Sikh and Not Sikh)
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    for cls in classes:
        # Create class directories in train_split and val
        os.makedirs(os.path.join(data_dir, 'train_split', cls), exist_ok=True)
        os.makedirs(os.path.join(data_dir, 'val', cls), exist_ok=True)
        
        # Get all images in the class
        class_dir = os.path.join(train_dir, cls)
        images = [img for img in os.listdir(class_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle images
        np.random.shuffle(images)
        
        # Split images into train and validation sets
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy images to train_split and val directories
        for img in train_images:
            src = os.path.join(train_dir, cls, img)
            dst = os.path.join(data_dir, 'train_split', cls, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        
        for img in val_images:
            src = os.path.join(train_dir, cls, img)
            dst = os.path.join(data_dir, 'val', cls, img)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
    
    return os.path.join(data_dir, 'train_split'), os.path.join(data_dir, 'val')

# Create train/val split
train_dir, val_dir = create_train_val_split(data_dir)

# Load datasets
image_datasets = {
    'train': datasets.ImageFolder(train_dir, data_transforms['train']),
    'val': datasets.ImageFolder(val_dir, data_transforms['val'])
}

# Create data loaders
dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=4, shuffle=True, num_workers=4
    ) for x in ['train', 'val']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(f"Classes: {class_names}")
print(f"Training samples: {dataset_sizes['train']}")
print(f"Validation samples: {dataset_sizes['val']}")

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """
    Trains the model and returns the best model based on validation accuracy.
    """
    since = time.time()
    
    # Deep copy of the best model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            
            # Iterate over data batches
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model if best accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

def visualize_model(model, num_images=6):
    """
    Visualizes model predictions on validation data.
    """
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(12, 8))
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}, actual: {class_names[labels[j]]}')
                
                # Convert tensor to image for display
                img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                ax.imshow(img)
                
                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.tight_layout()
                    plt.show()
                    return
        
        model.train(mode=was_training)
        plt.tight_layout()
        plt.show()

def save_model(model, filename='model.pth'):
    """
    Saves the model to a file.
    """
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def load_model(filename='model.pth', num_classes=2):
    """
    Loads a model from a file for inference.
    """
    model = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model.load_state_dict(torch.load(filename, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_image(model, image_path):
    """
    Predicts the class of an image using a trained model.
    """
    # Load and preprocess the image
    img = Image.open(image_path).convert('RGB')
    img_tensor = data_transforms['val'](img).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        
    return class_names[preds[0]]

# Load and set up the model
model = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model
model = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)

# Save the model
save_model(model, 'model.pth')

# Visualize some predictions
visualize_model(model)

# Example of inference
print("\nTesting model on test images:")
test_dir = os.path.join(data_dir, 'test')
test_images = [os.path.join(test_dir, img) for img in os.listdir(test_dir) 
              if img.endswith(('.jpg', '.jpeg', '.png'))]

for img_path in test_images:
    prediction = predict_image(model, img_path)
    print(f"Image: {os.path.basename(img_path)}, Prediction: {prediction}") 