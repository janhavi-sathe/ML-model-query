import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# 取得當前檔案的絕對路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cnn5layer_multi_label.pth")
TEST_NPZ = os.path.join(BASE_DIR, "X_test_image.npz")
image_prediction_done = False


# Custom CNN Model
class CNN5Layer(nn.Module):
    def __init__(self, num_classes=15):
        super(CNN5Layer, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # Layer 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Layer 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # Layer 3
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # Layer 4
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  # Layer 5

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Downsampling
        self.relu = nn.ReLU()  # Activation function

        self.fc = nn.Linear(512 * 7 * 7, num_classes)  # Fully Connected Layer
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = self.pool(self.relu(self.conv5(x)))

        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)  # No activation (raw logits for multi-label classification)
        return x

# Custom Dataset Class
class NPZImageDataset(torch.utils.data.Dataset):
    def __init__(self, npz_path, transform=None):
        data = np.load(npz_path, allow_pickle=True)  # 確保預先加載
        self.images = data['x']
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32) # 減少 I/O 操作
        
        if len(image.shape) == 2:  # Grayscale to 3 channels
            image = image.unsqueeze(0).repeat(3, 1, 1)
        else:
            image = image.permute(2, 0, 1)  # (H, W, C) -> (C, H, W)

        if self.transform:
            image = self.transform(image)
        return image


def predict_and_save():
    '''
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_dataset = NPZImageDataset(TEST_NPZ, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device("cpu")
    model = CNN5Layer(num_classes=10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    print("開始預測...")
    total_batches = len(test_loader)  # 計算總 batch 數

    # Perform predictions and save results
    predictions = []
    with torch.no_grad():
        for batch_idx, images in enumerate(test_loader, start=1):
            images = images.to(device)
            outputs = model(images)
            predictions.append(torch.sigmoid(outputs)) 
            if batch_idx % 5 == 0 or batch_idx == total_batches:
                print(f"Processing batch {batch_idx}/{total_batches}... ({(batch_idx / total_batches) * 100:.2f}%)")


    print("預測完成，儲存結果...")
    # Convert to numpy array and save
    predictions = torch.cat(predictions, dim=0).cpu().numpy()  # 最後一次轉 NumPy

    np.save(os.path.join(BASE_DIR, "y_pred_image.npy"), predictions)'''
    print(f"🎉 預測結果已儲存至y_pred_image")

    global image_prediction_done
    image_prediction_done = True
    print(f"image Prediction done: {image_prediction_done}")


