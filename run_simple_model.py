#!/usr/bin/env python3
"""
Simple script to run the ultra-low accuracy model we found
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, '.')

def create_simple_model():
    """Create a very simple CNN model for ultra-low accuracy"""
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            # Very simple architecture for low accuracy
            self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1)  # 32x32 -> 16x16
            self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)  # 16x16 -> 8x8
            self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)  # 8x8 -> 8x8
            self.pool = nn.AdaptiveAvgPool2d(1)  # 8x8 -> 1x1
            self.fc = nn.Linear(32, 10)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    return SimpleCNN()

def main():
    print("🚀 Running Ultra-Low Accuracy Model")
    print("=" * 50)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Create model
    model = create_simple_model().to(device)
    print(f"🏗️  Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Data preparation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Load CIFAR-10
    print("📦 Loading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    print("🎯 Starting training (expecting low accuracy)...")
    print("=" * 50)
    
    # Quick training loop (just a few epochs for demo)
    for epoch in range(3):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader):
            if i >= 100:  # Limit to 100 batches for speed
                break
                
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch {epoch + 1}: Loss: {running_loss / 100:.3f}, Accuracy: {accuracy:.1f}%')
    
    # Test accuracy
    print("\n🧪 Testing model...")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    final_accuracy = 100 * correct / total
    print(f"✅ Final Test Accuracy: {final_accuracy:.1f}%")
    
    if final_accuracy < 40:
        print("🎉 SUCCESS! Model has ultra-low accuracy (< 40%) as requested!")
    else:
        print("⚠️  Model accuracy is higher than expected, but still relatively low.")
    
    print(f"📊 Model Summary:")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - Test Accuracy: {final_accuracy:.1f}%")
    print(f"   - No LLM dependencies used")

if __name__ == "__main__":
    main()
