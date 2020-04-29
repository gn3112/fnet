from torchvision.models import vgg11
from torch import nn, optim
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
import torch
from torch import nn, optim
import argparse
from sklearn.metrics import confusion_matrix
import json
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Agg")

class Resize_AR():
    def __init__(self, size):
        self.size = size
        self.resize = transforms.Resize((size,size))

    def __call__(self, pil_image):
        h, w = pil_image.size
        if h > self.size and h > w:
            c = int((h-w)/2)
            pad = transforms.Pad((0,c),fill=(255,255,255))
            pil_image = pad(pil_image)
        elif w > self.size and w > h:
            c = int((w-h)/2)
            pad = transforms.Pad((c,0),fill=(255,255,255))
            pil_image = pad(pil_image)


        img = self.resize(pil_image)
        return img
        #     scale_by = self.size/h
        # elif w > self.size and w >= h:
        #     scale_by = self.size/w
        
        # pil_image = transforms.functional.affine(pil_image, 0, (0,0), scale_by, 0, fillcolor=(255,255,255))
        # h, w = pil_image.size
        # print(h,w)
        # if h < self.size:
        #     pad = transforms.Pad((0,self.size-h),fill=(255,255,255))
        #     pil_image = pad(pil_image)
        # elif w < self.size:
        #     pad = transforms.Pad((self.size-h,0),fill=(255,255,255))
        #     pil_image = pad(pil_image)

        return pil_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--hyperparam', required=True)

    args = parser.parse_args()
    with open(args.hyperparam) as f:
        hyperparam = json.load(f)

    EPOCHS = hyperparam['epochs']
    BATCH_SIZE = hyperparam['batch_size']
    LEARNING_R = hyperparam['learning_rate']
    # DEBUG = hyperparam['DEBUG']

    exp_path = "model_exp/"+hyperparam["exp_name"]

    writer = SummaryWriter()

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T = transforms.Compose([Resize_AR(224), transforms.ToTensor()])

    model = vgg11(pretrained=True)

    for param1, param2 in zip(model.features.parameters(), model.avgpool.parameters()):
        param1.requires_grad = False
        param2.requires_grad = False

    model.classifier[-1] = nn.Linear(4096, 125)
    model.classifier[2] = nn.Dropout(p=0.25)
    model.classifier[5] = nn.Dropout(p=0.25)

    model = model.to(device)

    training_data = datasets.ImageFolder(root="/home/georgesnomicos/bigbird_s_view/", transform=T)

    indices = list(range(len(training_data)))
    split = int(np.floor(0.1 * len(training_data)))
    np.random.seed(40)
    np.random.shuffle(indices)
    train_indices, valid_indices = indices[split:], indices[:split]

    train_size = len(train_indices)
    valid_size = len(valid_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = DataLoader(training_data, batch_size=BATCH_SIZE, sampler=train_sampler)
    valid_loader = DataLoader(training_data, batch_size=BATCH_SIZE, sampler=valid_sampler)

    optimizer = optim.RMSprop([{"params": model.classifier[0:6].parameters()},
        {"params": model.classifier[-1].parameters(), 'lr':LEARNING_R*10}], 
        lr=LEARNING_R)

    exp_lr_decay = optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    softmax = nn.Softmax(dim=1)
    count = 0
    for epoch in range(EPOCHS):
        print("Epoch {}/{}".format(epoch+1, EPOCHS))
        print("-" * 10)
        running_loss_train = 0
        running_correct_train = 0
        for inputs, labels in train_loader:
            labels = labels.to(device)

            optimizer.zero_grad()

            out = model(inputs.to(device))

            loss = criterion(out, labels)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                _, pred = torch.max(softmax(out), 1)

            running_correct_train += torch.sum(pred==labels)

            running_loss_train += loss.item() * inputs.size(0)

            
            count += 1

        epoch_loss_train = running_loss_train/train_size
        epoch_accuracy_train = running_correct_train.type(torch.DoubleTensor)/train_size

        writer.add_scalar('Loss/Train',epoch_loss_train, epoch)
        writer.add_scalar('Accuracy/Train', epoch_accuracy_train, epoch)
        
        print("Training, Loss:{} Accuracy: {} Lr: {}".format(epoch_loss_train,epoch_accuracy_train, exp_lr_decay.get_lr()[0]))

        exp_lr_decay.step()

        if epoch != 0 and epoch % 25 == 0:
            torch.save(model.state_dict(), os.path.join(exp_path,"model_{}.pt".format(epoch)))
            writer.flush()

    y_true = []
    y_pred = []

    idx_p = 0
    running_correct_test = 0
    fail_examples = {}
    idx_to_name = {values: keys for keys, values in training_data.class_to_idx.items()}

    model.eval()
    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            out = model(inputs)

        probas, pred = torch.max(softmax(out),1)
        
        y_true = y_true + labels.tolist()
        y_pred = y_pred + pred.tolist()

        running_correct_test += torch.sum(pred==labels)
        for i in range(len(labels.tolist())):
            if labels.tolist()[i] != pred.tolist()[i]:
                # save_image(inputs[i,:,:,:],os.path.join(exp_path,"fail_{}.jpeg".format(idx_p)))
                fail_examples = {'img_file_name':"fail_{}.jpeg".format(idx_p),
                        'class':idx_to_name[labels.tolist()[i]],
                        'predicted':idx_to_name[pred.tolist()[i]],
                        'confidence':round(probas[i].item(),3)}
                idx_p += 1

                writer.add_image("Fail_Images/fail{}".format(idx_p),inputs[i,:,:,:].view(3,224,224))
                writer.add_text("Fail_meta/fail{}".format(idx_p), str(fail_examples))

    # writer.add_graph(model, input_to_model=inputs[-1,:,:,:].view(1,3,224,224))

    accuracy_test = running_correct_test.type(torch.DoubleTensor)/valid_size

    cf = confusion_matrix(y_true, y_pred)
    class_idx = y_pred + y_true

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
    # plt.savefig(os.path.join(exp_path,"cf.jpeg"))
    writer.add_figure("Confusion Matrix", fig)

    print("Test, Accuracy: {}".format(accuracy_test))

    # with open(os.path.join(exp_path,"fail_examples.txt"),'w') as f:
    #     json.dump(fail_examples, f)

    # Saving parameters network and create reload and test function
    torch.save(model.state_dict(), os.path.join(exp_path,"model.pt"))

    writer.add_hparams(hyperparam,{'hparam/accuracy': accuracy_test})
    writer.close()
    # img = T(Image.open("/Users/georgesnomicos/fridge_net/industrial_items/Beurre_gastronomique_doux_PRESIDENT.jpg"))

    # log = model(img.view(-1,3,224,224))
    # softmax = nn.Softmax()
    # print(torch.max(softmax(log)))