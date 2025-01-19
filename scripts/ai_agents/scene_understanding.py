import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report, accuracy_score
import cv2
import numpy as np

# Parameters
num_classes = 13  # Number of classes in UCF Crime dataset
sequence_length = 16  # Number of frames per sequence
input_size = 224  # Input resolution for images
batch_size = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset Class
class UCFCrimeImageSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Path to train or test directory.
            transform (callable, optional): Transformations for the frames.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.samples = []
        for label, class_dir in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_dir)
            sequences = sorted(os.listdir(class_path))
            for seq in sequences:
                seq_path = os.path.join(class_path, seq)
                self.samples.append((seq_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq_path, label = self.samples[idx]
        frames = self._load_frames(seq_path)
        return frames, label

    def _load_frames(self, folder_path):
        frames = []
        frame_files = sorted(os.listdir(folder_path))[:sequence_length]
        for frame_file in frame_files:
            frame_path = os.path.join(folder_path, frame_file)
            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, (input_size, input_size))
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        # Stack frames and pad if necessary
        frames = np.stack(frames, axis=0) if len(frames) == sequence_length else np.zeros((sequence_length, 3, input_size, input_size))
        return torch.tensor(frames, dtype=torch.float32)

# Define Model: Simple 3D CNN
class Simple3DCNN(nn.Module):
    def __init__(self, num_classes):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.fc1 = nn.Linear(128 * (sequence_length // 1) * (input_size // 4) * (input_size // 4), 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset and DataLoaders
train_dir = "Path/to/train"
test_dir = "path/to/test"

train_dataset = UCFCrimeImageSequenceDataset(train_dir, transform=transform)
test_dataset = UCFCrimeImageSequenceDataset(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model, Loss, Optimizer
model = Simple3DCNN(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.permute(0, 4, 1, 2, 3)  # Rearrange to [batch, channels, sequence, height, width]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

print("Training complete.")

# Evaluation
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        inputs = inputs.permute(0, 4, 1, 2, 3)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Metrics
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=train_dataset.classes))
