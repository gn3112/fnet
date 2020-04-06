from fruit_model import Fruit_Classifier
from PIL import Image
from torchvision import transforms
import torch
import io
import json

data_transform = transforms.Compose([
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
    stream = io.BytesIO(img)
    img = Image.open(stream)
    img = data_transform(img)
    prob, idx = torch.max(torch.exp(net(img.view(-1,3,100,100))),1)
    return prob, [key for key, value in class_idx.items() if value == idx]
