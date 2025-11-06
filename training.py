# I have updated the learning rate(1e-4 -> 5e-5) and no. of epochs (20 -> 30) after acheiving 96.99% accuracy 
# also removed augmentation for the later epochs ran after 96.99% accuracy 
# currently best accuracy acheived is 97.5%
import torch
import torch.nn as nn
import torch.optim as optim
import os
import glob
import random
import numpy as np
from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm 

ROI_ROOT = 'C:\\Users\\Lenovo\\PCB Python\\final_defect_roi_1' 

IMAGE_SIZE = 128          
BATCH_SIZE = 32         
NUM_EPOCHS = 30      
LEARNING_RATE = 5e-5      
MODEL_NAME = 'efficientnet_b4' 
MODEL_CHECKPOINT_PATH = 'efficientnet_b4_pcb_classifier_best.pth' 

# Set seeds for reproducibility (good practice)
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class DefectROIDataset(Dataset):
    """
    Custom Dataset class to correctly handle the two-level directory structure 
    by mapping all images to the parent Defect_Type folder.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        #  Find all image files using recursion
        self.all_files = []
        for ext in ('*.jpg', '*.png'):
            self.all_files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        
        # Extract Class Names (Defect Type) from the path
        class_names = set()
        for filepath in self.all_files:
            relative_path = os.path.relpath(filepath, root_dir)
            if os.path.sep in relative_path:
                # Takes the first folder name as the defect type
                defect_type = relative_path.split(os.path.sep)[0]
                class_names.add(defect_type)
        
        # Create numerical label mappings
        self.class_to_idx = {name: i for i, name in enumerate(sorted(list(class_names)))}
        self.idx_to_class = {i: name for name, i in self.class_to_idx.items()}
        
        #Final list of (filepath, label_index) pairs
        self.file_label_pairs = []
        for filepath in self.all_files:
            relative_path = os.path.relpath(filepath, root_dir)
            if os.path.sep in relative_path:
                defect_type = relative_path.split(os.path.sep)[0]
                label = self.class_to_idx[defect_type]
                self.file_label_pairs.append((filepath, label))
                
        if not self.file_label_pairs:
             raise FileNotFoundError(f"No labeled images found in subdirectories of {root_dir}. Check the path and folder structure.")

    def __len__(self):
        return len(self.file_label_pairs)

    def __getitem__(self, idx):
        img_path, label = self.file_label_pairs[idx]
        
        # Load image and convert to RGB
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

    @property
    def classes(self):
        return [self.idx_to_class[i] for i in range(len(self.idx_to_class))]


# DATA LOADERS

def create_dataloaders(root_dir, img_size, batch_size):
    
    # Standard normalization parameters for models pre-trained on ImageNet
    NORM_MEAN = [0.485, 0.456, 0.406]
    NORM_STD = [0.229, 0.224, 0.225]
    
    # Define transformations for preprocessing and augmentation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),

            #transforms.RandomRotation(15),           
            #transforms.ColorJitter(brightness=0.1),  

            transforms.ToTensor(),                   
            transforms.Normalize(NORM_MEAN, NORM_STD)
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD) 
        ]),
    }

    # Use the custom dataset to load and label the data
    full_dataset = DefectROIDataset(root_dir)
    
    # Split dataset into training (80%) and validation (20%) sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(full_dataset)), [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # Apply transforms to the subsets
    train_subset.dataset.transform = data_transforms['train']
    val_subset.dataset.transform = data_transforms['val']

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, len(full_dataset.classes), full_dataset.classes


# MODEL, LOSS, AND OPTIMIZER SETUP 

def setup_model(num_classes, model_name, checkpoint_path=None):
    
    # Load EfficientNet-B4 model with pre-trained weights
    model = timm.create_model(model_name, pretrained=True)
    
    #Transfer Learning
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    # RESUME LOGIC
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path} to resume training...")
        # Load the saved state dictionary
        try:
            model.load_state_dict(torch.load(checkpoint_path))
            print("Checkpoint loaded successfully. Resuming training.")
        except RuntimeError as e:
            print(f"Error loading checkpoint: {e}")
            print("Model architecture may have changed, starting training from scratch on top of pre-trained weights")

    # Cross-Entropy Loss  and Adam optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    
    return model, criterion, optimizer


#TRAINING LOOP

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, checkpoint_path):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Check if a checkpoint exists to set the starting best accuracy
    best_accuracy = 0.0
    if os.path.exists(checkpoint_path):
        #Run one validation epoch to get the starting accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        best_accuracy = 100 * correct / total
        print(f"Resuming training. Initial accuracy from checkpoint is: {best_accuracy:.2f}%")


    print(f"Starting training on device: {device}...")
    
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        running_loss = 0.0
        
        #Training Phase
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() 
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward() 
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)

        #Validation Phase
        model.eval() # Set model to evaluation mode
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Val Accuracy: {epoch_accuracy:.2f}%")
        
        # Checkpoint: Saving the best model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"*** Model checkpoint saved! New best accuracy: {best_accuracy:.2f}% ***")
            
    print(f"\nTraining finished. Final best validation accuracy: {best_accuracy:.2f}% (Target: 97%)")


# EXECUTION

if __name__ == "__main__":
    if ROI_ROOT.startswith('path/to/'):
        print("!!! ACTION REQUIRED: UPDATE ROI_ROOT PATH WITH YOUR DATA !!!")

    else:
        train_loader, val_loader, num_classes, class_names = create_dataloaders(ROI_ROOT, IMAGE_SIZE, BATCH_SIZE)
        print(f"Data loaded successfully. Found {num_classes} defect classes: {class_names}")
        
        model, criterion, optimizer = setup_model(num_classes, MODEL_NAME, MODEL_CHECKPOINT_PATH)
        print(f"Model {MODEL_NAME} configured for {num_classes} classes.")
        

        train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, MODEL_CHECKPOINT_PATH)


