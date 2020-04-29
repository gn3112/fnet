from torchvision.models import vgg11
import torch
from PIL import Image
import os
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from vgg_single_view_detection import Resize_AR
from torch import nn, optim
from torch.utils.data import DataLoader

class ItemDataset():
    def __init__(self, PATH, transform=None):
        self.PATH = PATH
        if not transform:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform
        self.samples_file = [i for i in os.listdir(PATH) if not i.startswith(".")]

        self.idx_to_class = self._idx_to_class()
        self.class_to_idx = {class_:idx for idx, class_ in self.idx_to_class.items()}


    def __len__(self):
        return len(self.idx_to_class)
    
    def _idx_to_class(self):
        idx_to_class = {}
        for idx, class_name in enumerate(self.samples_file):
            idx_to_class[idx] = class_name.strip(".jpg")
        
        return idx_to_class

    def __getitem__(self, idx):
        sample_file = self.samples_file[idx]
        img_PATH = os.path.join(self.PATH, sample_file)

        img = Image.open(img_PATH)
        img = self.transform(img)

        sample_class_name = sample_file.strip(".jpg")

        sample = {"image": img, "label": torch.tensor(self.class_to_idx[sample_class_name])}

        return sample


if __name__ == "__main__":
    MODEL_PATH = "/Users/georgesnomicos/fridge_net/model_exp/vgg_3_dropout0.25_lrstep35/model.pt"
    DATASET_PATH = "/Users/georgesnomicos/fridge_net/industrial_items"
    LEARNING_R = 0.001
    BATCH_SIZE = 8
    EPOCHS = 50
    EXP_NAME = "VGG_EXP/VGG_Single_Training_vgg3"

    TEST_DIR = "/Users/georgesnomicos/fridge_net/testing_Beurre_gastronomique_doux_PRESIDENT"

    writer = SummaryWriter(log_dir=EXP_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = vgg11()
    model.classifier[-1] = nn.Linear(4096, 125)
    model.classifier[2] = nn.Dropout(p=0.25)
    model.classifier[5] = nn.Dropout(p=0.25)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    T = transforms.Compose([Resize_AR(224), transforms.ToTensor()])

    dataset = ItemDataset(DATASET_PATH, transform=T)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=True, num_workers=0)

    model.classifier[-1] = nn.Linear(4096, len(dataset.class_to_idx))

    model = model.to(device)

    for param1, param2 in zip(model.features.parameters(), model.avgpool.parameters()):
        param1.requires_grad = False
        param2.requires_grad = False

    optimizer = optim.RMSprop([{"params": model.classifier[0:6].parameters()},
    {"params": model.classifier[-1].parameters(), 'lr':LEARNING_R*10}], 
    lr=LEARNING_R)

    exp_lr_decay = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    softmax = nn.Softmax(dim=1)

    print(dataset.class_to_idx)

    for epoch in range(EPOCHS):
        running_loss_train = 0
        running_correct_train = 0
        print("Epoch {}/{}".format(epoch+1, EPOCHS))
        print("-" * 10)
        model.train()
        for sample in dataloader:
            inputs = sample["image"].to(device)
            labels = sample["label"].to(device)
            
            optimizer.zero_grad()
            
            out = model(inputs)
            
            loss = criterion(out, labels)

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                proba, pred = torch.max(softmax(out), 1)

            print(proba)

            running_correct_train += torch.sum(pred==labels).item()

            running_loss_train += loss.item() * inputs.size(0)

        epoch_loss_train = running_loss_train/len(dataset)
        epoch_accuracy_train = float(running_correct_train/len(dataset))

        writer.add_scalar('Loss/Train',epoch_loss_train, epoch)
        writer.add_scalar('Accuracy/Train', epoch_accuracy_train, epoch)
        
        print("Training, Loss:{} Accuracy: {} Lr: {}".format(epoch_loss_train,epoch_accuracy_train, exp_lr_decay.get_lr()[0]))

        exp_lr_decay.step()

        model.eval()
        test_files = [i for i in os.listdir(TEST_DIR) if not i.startswith(".")]
        running_correct_test = 0
        for test_file in test_files:
            img = Image.open(os.path.join(TEST_DIR,test_file))
            with torch.no_grad():
                out = model(T(img).view(1,3,224,224))
            
            proba, pred = torch.max(softmax(out), 1)
            print(proba, dataset.idx_to_class[pred.item()])
            running_correct_test += torch.sum(pred==torch.tensor([4])).item()
        
        test_accuracy = float(running_correct_test/len(test_files))

        writer.add_scalar('Accuracy/Test-Butter', test_accuracy, epoch)
        print("Test, Accuracy: {}".format(test_accuracy))
