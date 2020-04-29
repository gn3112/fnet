import cv2
import numpy as np
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor
import os

def create_dir_exp(exp_type, exp_name): 
    exp_types = ["VGG_EXP", "model_exp"]
    for exp in exp_types:
        if exp in exp_type
            exp_type_correct = True

    if not exp_type_correct:
        raise NameError("Please specify correct exp_type.")

    final_path = os.path.join("EXP", exp_type, exp_name)

    if os.path.exists(final_path):
        raise NameError("Experiment {} already exists.".format(exp_name))

    return final_path

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