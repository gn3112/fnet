import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

class Fruit_Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride=1, padding=2)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)
        self.conv4 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.maxpool4 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(5*5*128, 1024)
        self.fc2 = nn.Linear(1024, 120)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # save_image(x, "test.png")
        x = self.maxpool1(F.relu(self.conv1(x)))
        x = self.maxpool2(F.relu(self.conv2(x)))
        x = self.maxpool3(F.relu(self.conv3(x)))
        x = self.maxpool4(F.relu(self.conv4(x)))
        x = F.relu(self.fc1(x.view(-1,5*5*128)))
        x = F.relu(self.fc2(x))
        return self.softmax(x)

class to_hsv(object):
    def __call__(self, img):
        return img.convert('HSV')

if __name__ == "__main__":
    EPOCHS = 10
    device = torch.device("cuda")

    fruit_selection = ["Banana", "Tomato", "Onion", "Avocado", "Clementine", "Mandarine"]
    fruit_names = os.listdir("fruits-360_dataset/fruits-360/Training/")

    data_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0, contrast=0, saturation=[0.9,1.2], hue=0.02),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.Resize(100),
        transforms.ToTensor()
        ])

    training_data = datasets.ImageFolder(root="fruits-360_dataset/fruits-360/Training/", transform=data_transform)
    max = 0

    indices = list(range(len(training_data)))
    split = int(np.floor(0.15 * len(training_data)))
    np.random.seed(40)
    np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]

    train_size = len(train_indices)
    valid_size = len(valid_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = DataLoader(training_data, batch_size=64, sampler=train_sampler)
    valid_loader = DataLoader(training_data, batch_size=64, sampler=valid_sampler)

    net = Fruit_Classifier().to(device)

    # Optimisation
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    exp_lr_decay = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    criterion = nn.NLLLoss()

    for epoch in range(EPOCHS):
        print("Epoch {}/{}".format(epoch+1, EPOCHS))
        print("-" * 10)
        running_loss_train = 0
        running_correct_train = 0
        running_loss_valid = 0
        running_correct_valid = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            out = net(inputs)
            loss = criterion(out, labels)
            _, pred = torch.max(torch.exp(out),1)
            loss.backward()
            optimizer.step()
            # print(net.fc2.weight)
            running_loss_train += loss.item() * inputs.size(0)
            running_correct_train += torch.sum(pred==labels)

        exp_lr_decay.step()

        net.eval()

        for inputs, labels in valid_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            out = net(inputs)
            loss = criterion(out, labels)
            _, pred = torch.max(torch.exp(out),1)
            running_loss_valid += loss.item() * inputs.size(0)
            running_correct_valid += torch.sum(pred==labels)

        net.train()

        epoch_loss_train = running_loss_train/train_size
        epoch_accuracy_train = torch.tensor(running_correct_train).type(torch.DoubleTensor)/train_size
        epoch_loss_valid = running_loss_valid/valid_size
        epoch_accuracy_valid = torch.tensor(running_correct_valid).type(torch.DoubleTensor)/valid_size

        print("Training, Loss:{} Accuracy: {}".format(epoch_loss_train,epoch_accuracy_train))
        print("Validation, Loss:{} Accuracy: {}".format(epoch_loss_valid,epoch_accuracy_valid))

    # Testing
    data_transform = transforms.Compose([
        transforms.Resize(100),
        transforms.ToTensor()
        ])

    net.eval()
    test_data = datasets.ImageFolder(root="fruits-360_dataset/fruits-360/Test/", transform=data_transform)
    test_loader = DataLoader(test_data, batch_size=128)
    running_correct_test=0
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        out = net(inputs)

        _, pred = torch.max(torch.exp(out),1)

        running_correct_test += torch.sum(pred==labels)

    accuracy_test = torch.tensor(running_correct_test).type(torch.DoubleTensor)/len(test_data)

    print("Test, Accuracy: {}".format(accuracy_test))


    # Saving parameters network and create reload and test function
    torch.save(net.state_dict(),"model.pt")

    # More criterions and fail example

    img = training_data[1][0].numpy()
    plt.imshow(np.transpose(img, [1,2,0]))
    plt.show()
