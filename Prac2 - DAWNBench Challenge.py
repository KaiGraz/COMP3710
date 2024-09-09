from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
import torchvision
from torchvision import transforms

# With support from Shakes (Summer of AI 2022):
# https://www.youtube.com/watch?v=pvwxIOIqgmc&list=PLKB59ot0pqdU-VbpdkKgNLXuHx2OramTt

########## Device Selection ##########
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# if device == "cpu":
#     print("Using CPU, not CUDA")
print(f"Using {device} device")



########## Data Prep ##########

batch_size = 128
cifar_means = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2023, 0.1994, 0.2010)

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_means, cifar_std),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4, padding_mode="reflect")
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_means, cifar_std)
])

train_set = torchvision.datasets.CIFAR10(
    root="cifar10", train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.CIFAR10(
    root="cifar10", train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

print("Data Prepped")

########## Resnet ##########

class ResNet_Block(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, kernel_size=3):
        super(ResNet_Block, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_planes,planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(
            planes,planes, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))
            
    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out
        
class ResNet_Model(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, stride=1, kernel_size=3) -> None:
        super(ResNet_Model, self).__init__()
        
        # Hyperparameters
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(
            3, self.in_planes, kernel_size=kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        
        self.layer1 = self._make_layer(block, self.in_planes, num_blocks[0], stride=stride)
        self.layer2 = self._make_layer(block, 2*self.in_planes, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 4*self.in_planes, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 8*self.in_planes, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(self.in_planes*block.expansion, num_classes)
        
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet(num_epochs=10, lr = 0.1, momentum = 0.9, weight_decay = 5e-4, verbose=True):
    model = ResNet_Model(ResNet_Block, [2, 2, 2, 2])
    model = model.to(device)
    
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        
    print(f"Model No. of Parameters: {sum([param.nelement() for param in model.parameters()])}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    # Piecewise Linear Schedule (from Shakes)
    total_step = len(train_loader)
    sched_linear_1 = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=0.005, max_lr=lr, step_size_up=15, step_size_down=15,
        mode="triangular")
    sched_linear_3 = torch.optim.lr_scheduler.LinearLR(
        optimizer,start_factor=0.005/lr, end_factor=0.005/5)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[sched_linear_1, sched_linear_3], milestones=[30])
    
    
    
    # Train the model
    model.train()
    print("Training")
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backwards and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i+1) % 100 == 0:
                print(f"Epoch[{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.5f}")
        scheduler.step()
    
    print("Training done, now Testing")
    
    # Test the model
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
        
        
            _, predicted_labels = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct = (predicted_labels == labels).sum().item()
        
        if verbose:
            print("Total Testing:", total)
            print("Predictions:", predicted_labels.numpy())
            print("Which Correct:", correct)
            print("Total Correct:", np.sum(correct))
            print("Accuracy:", np.sum(correct) / total)
        

ResNet(num_epochs=2)