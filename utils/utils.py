import cv2
import numpy as np
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor
def cv_format(img):
    img = img * 255
    img = img.transpose(1, 2, 0)
    img = img.astype(np.uint8) 
    return img

def show_bbox(img, bboxes):
    img = cv_format(img)

    for bbox in bboxes:
        bbox = [int(i) for i in bbox]
        img = cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2], bbox[3]),2,2)

    cv2.imshow("",img)
    cv2.waitKey(0)

def overlap_feature(feature, img):
    img = cv_format(img)
    heatmap_img = cv_format(feature)
    overlap_img = cv2.addWeighted(heatmap_img, 0.7, img, 0.3, 0)
    cv2.imshow("",overlap_img)
    cv2.waitKey(0)

def transform(im):
    T = Compose([ToPILImage(), Resize((256,256)), ToTensor()])
    return T(im)