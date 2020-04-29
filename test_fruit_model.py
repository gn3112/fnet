from fruit_model import Fruit_Classifier
from PIL import Image
from torchvision import transforms
import torch
import io
import json
import os

data_transform = transforms.Compose([
    transforms.CenterCrop(100),
    transforms.Resize(100),
    transforms.ToTensor()
    ])

device = torch.device('cpu')

with open('class_idx.txt') as f:
    class_idx = json.load(f)

def fruit_test(img):
    net = Fruit_Classifier()
    net.load_state_dict(torch.load('model.pt', map_location=torch.device('cpu')))
    net.eval()
    if type(img) == str:
        img = Image.open(img)
    else:
        stream = io.BytesIO(img)
        img = Image.open(stream)
    img = data_transform(img)
    with torch.no_grad():
        prob, idx = torch.max(torch.exp(net(img.view(-1,3,100,100))),1)
    return prob, [key for key, value in class_idx.items() if value == idx]

# path = 'fruits-360_dataset/fruits-360/test-multiple_fruits/'
# path = 'new_fruit_dataset/pi_data/'
# imgs = os.listdir(path)
path = ""
imgs = ['apple.jpg']
for img in imgs:
    img_path = path + img
    if img[:3] == "063":
        continue
    else:
        fruit_name = [key for key, value in class_idx.items() if value == img[:3]]
        print("Predicted: ", fruit_test(img_path)," ","Actual: ",fruit_name, " ", img)