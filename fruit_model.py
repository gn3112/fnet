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
import json
from sklearn.metrics import confusion_matrix
from visdom import Visdom

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
        self.fc2 = nn.Linear(1024, 63)
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

# Extra Transforms:
class to_hsv(object):
    def __call__(self, img):
        return img.convert('HSV')

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparam', required=True)

    args = parser.parse_args()
    print(args.hyperparam)
    with open(args.hyperparam) as f:
        hyperparam = json.load(f)

    EPOCHS = hyperparam['epochs']
    BATCH_SIZE = hyperparam['batch_size']
    LEARNING_R = hyperparam['learning_rate']
    DEBUG = hyperparam['DEBUG']

    exp_path = "model_exp/"+hyperparam["exp_name"]
    
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        viz = Visdom()
    except:
        print('Could not find Visdom server')
        pass

    fruit_selection = ["Banana", "Tomato", "Onion", "Avocado", "Clementine", "Mandarine"]
    fruit_names = os.listdir("new_fruit_dataset/Training/")

    data_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0, contrast=0, saturation=[0.9,1.2], hue=0.02),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomApply([transforms.RandomAffine(180,fillcolor=(255,255,255))],p=1),
        transforms.RandomApply([transforms.RandomAffine(0,fillcolor=(255,255,255), scale=(0.4,1))],p=0.4),
        transforms.Resize(100),
        transforms.ToTensor(),
        transforms.RandomApply([AddGaussianNoise()],p=0.4),
        ])

    training_data = datasets.ImageFolder(root="new_fruit_dataset/Training/", transform=data_transform)
    # with open('class_idx.txt', 'w') as outfile:
    #     json.dump(training_data.class_to_idx, outfile)

    indices = list(range(len(training_data)))
    split = int(np.floor(0.15 * len(training_data)))
    np.random.seed(40)
    np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]

    train_size = len(train_indices)
    valid_size = len(valid_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, sampler=train_sampler)
    valid_loader = DataLoader(training_data, batch_size=BATCH_SIZE, sampler=valid_sampler)

    net = Fruit_Classifier().to(device)

    # Optimisation
    optimizer = optim.SGD(net.parameters(), lr=LEARNING_R)
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

            if DEBUG:
                break

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
            if DEBUG:
                break
        net.train()

        epoch_loss_train = running_loss_train/train_size
        epoch_accuracy_train = running_correct_train.type(torch.DoubleTensor)/train_size
        epoch_loss_valid = running_loss_valid/valid_size
        epoch_accuracy_valid = running_correct_valid.type(torch.DoubleTensor)/valid_size

        try:
            if epoch == 0:
                loss_viz = viz.line(X=np.array([float(epoch)]),
                Y=np.column_stack(([epoch_loss_train],[epoch_loss_valid])),
                opts=dict(xlabel='Epoch',
                legend=['train','valid'],
                ylabel='Loss',
                title='Train and Valid'))
            else:
                viz.line(X=np.array([float(epoch)]),
                    Y=np.column_stack(([epoch_loss_train],[epoch_loss_valid])),
                    win=loss_viz,
                    update='append')
        except:
            pass
        
        print("Training, Loss:{} Accuracy: {}".format(epoch_loss_train,epoch_accuracy_train))
        print("Validation, Loss:{} Accuracy: {}".format(epoch_loss_valid,epoch_accuracy_valid))
        if DEBUG:
            break
    # Testing
    data_transform = transforms.Compose([
        transforms.Resize(100),
        transforms.ToTensor()
        ])

    net.eval()
    test_data = datasets.ImageFolder(root="new_fruit_dataset/Test", transform=data_transform)
    test_loader = DataLoader(test_data, batch_size=128)
    running_correct_test=0

    y_true = []
    y_pred = []
    
    idx_p = 0
    fail_examples = {}
    idx_to_name = {values: keys for keys, values in training_data.class_to_idx.items()}


    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        out = net(inputs)

        probas, pred = torch.max(torch.exp(out),1)
        
        y_true.append(labels.tolist())
        y_pred.append(pred.tolist())

        running_correct_test += torch.sum(pred==labels)

        if DEBUG:
            accuracy_test = 0 
            break
        else:
            accuracy_test = running_correct_test.type(torch.DoubleTensor)/len(test_data)

        for i in range(len(y_pred[0])):
            if y_pred[0][i] != y_true[0][i]:
                save_image(inputs[i,:,:,:],os.path.join(exp_path,"fail_{}.jpeg".format(idx_p)))
                fail_examples[idx_p] = {'img_file_name':"fail_{}.jpeg".format(idx_p),
                        'class':idx_to_name[y_true[0][i]],
                        'predicted':idx_to_name[y_pred[0][i]],
                        'confidence':round(probas[i].item(),3)}
                idx_p += 1

    cf = confusion_matrix(y_true[0], y_pred[0])
    class_idx = y_pred[0] + y_true[0]

    class_idx = list(dict.fromkeys(class_idx))
    class_idx.sort()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cf)
    fig.colorbar(cax)

    name_items = [idx_to_name[i] for i in class_idx]
    ax.set_xticks(np.arange(len(name_items)))
    ax.set_yticks(np.arange(len(name_items)))
    ax.set_xticklabels(name_items,rotation=0)
    ax.set_yticklabels(name_items,rotation=0)
    ax.set_xlabel('True Label')
    ax.set_ylabel('Predicated Label')
    for i in range(len(class_idx)):
        for j in range(len(class_idx)):
            text = ax.text(j, i, cf[i, j],
                        ha="center", va="center", color="w")
    plt.savefig(os.path.join(exp_path,"cf.jpeg"))

    print("Test, Accuracy: {}".format(accuracy_test))

    with open(os.path.join(exp_path,"fail_examples.txt"),'w') as f:
        json.dump(fail_examples, f)

    # Saving parameters network and create reload and test function
    if not DEBUG:
        torch.save(net.state_dict(), os.path.join(exp_path,"model.pt"))

    # img = training_data[1][0].numpy()
    # plt.imshow(np.transpose(img, [1,2,0]))
    # plt.show()
