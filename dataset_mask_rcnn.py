from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import cv2
from detectron2.structures import BoxMode

from pycocotools.mask import encode, toBbox


from torchvision import transforms

from math import sqrt
import random
import os
import json
import time

with open('class_idx.txt') as f:
    class_idx = json.load(f)

class_idx = {name: i for i, name in enumerate(['Tomato', 'Orange', 'Lemon', 'Apple', 'Banana', 'Avocado'])} 

def get_class_label(filename):
  pass

def get_bounding_box(img, mask):
    img = np.array(Image.fromarray(img.astype(np.uint8)).convert('L'))
    size = np.shape(img)
    thresh = 220
    min_x, min_y, max_x, max_y = 0,0,0,0
    # max y
    for i in range(size[0]):
        if np.amin(img[i,:]) < thresh:
            max_y = i
            break
    # min y
    for i in range(size[0]-1,0,-1):
        if np.amin(img[i,:]) < thresh:
            min_y = i
            break

    # min x
    for i in range(size[1]-1,0,-1):
        if np.amin(img[:,i]) < thresh:
            max_x = i
            break

    # max x
    for i in range(size[1]):
        if np.amin(img[:,i]) < thresh:
            min_x = i
            break      

    return [min_x, min_y, max_x, max_y], toBbox(encode(np.asarray(mask,dtype=np.uint8, order="F")))

def get_mask(img):
    img_grey = Image.fromarray(img.astype(np.uint8)).convert('L')
    img_grey = np.array(img_grey)
    mask_obj = (img_grey < 250)        
    return mask_obj 

def get_fruit_dicts_DEPR(img):
    dataset_dicts = []

    # img = cv2.imread("./10_100.jpg")
    # img = cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2], bbox[3]), (255,0,0))
    # cv2_imshow(img)

    record = {}
    objs = []

    record['file_name'] = "/Users/georgesnomicos/new_fruit_dataset/Training/Banana/10_100.jpg"
    record['image_id'] = 0
    record["height"] = 100
    record["width"] = 100

    coco_rle = pycocotools.mask.encode(mask_map.astype(np.uint8), order="F")
    print(coco_rle)
    obj = {
        "bbox": bbox,
        "bbox_mode": BoxMode.XYXY_ABS,
        "segmentation": coco_rle,
        "category_id": 0,
        "iscrowd": 0
    }

    objs.append(obj)

    record['annotations'] = objs

    dataset_dicts.append(record)

    return dataset_dicts

# import N number of image 
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 10)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def apply_texture(img, masks):
    path_textures = "/Users/georgesnomicos/dtd/images"
    texture_type = random.choice([f for f in os.listdir(path_textures) if not f.startswith('.')])
    texture_file = random.choice([f for f in os.listdir(os.path.join(path_textures, texture_type)) if not f.startswith('.')])

    img_texture = Image.open(os.path.join(path_textures,texture_type,texture_file)).convert('RGB')
    img_texture = img_texture.resize((256,256),Image.ANTIALIAS)
    img_texture = np.array(img_texture)

    mask_super = masks[0]

    for i in range(len(masks)-1):
        overlay = (masks[i+1] == 1)

        mask_super[overlay] = masks[i+1][overlay]

    mask_super = np.logical_not(mask_super)    

    img_with_texture = np.copy(img)
    img_with_texture[mask_super,:] = img_texture[mask_super,:]
    return img_with_texture

def generate_sample(prefix_name=None):
    CLS_PATH = "/Users/georgesnomicos/fridge_net/new_fruit_dataset/Training"
    
    classes = [f for f in os.listdir(CLS_PATH) if not f.startswith('.')]
    selected_classes = ['Tomato', 'Orange', 'Lemon', 'Apple', 'Banana', 'Avocado']

    n_objects = random.randint(1,3)
    masks = []
    bboxes = []
    all_img_cls = []
    all_img_file = []
    all_img = []
    location = []

    if not prefix_name:
        id = str(random.randint(0,1000))
        name =  id + '.jpeg'
        label_name = id + ".txt"
    else:
        id = int(prefix_name) 
        name = str(prefix_name) + '.jpeg'
        label_name = str(prefix_name) + '.txt'
    
    all_scale = []
    for n in range(n_objects):
        scale = round(random.uniform(0.7,1.2), 1)
        padding_sides = int((256 - 100*scale) / 2)
        
        all_scale.append(scale)

        counter = 0
        limit_reached = False

        while 1:
            counter += 1

            if counter > 200:
                limit_reached = True
                break

            tx = random.randint(-75, 75)
            ty = random.randint(-75, 75)

            if not location:
                break 
            
            all_dist = []
            for idx, txs_other in enumerate(location):
                tx_other, ty_other = txs_other
                dist = sqrt((tx - tx_other)**2 + (ty - ty_other)**2)
                all_dist.append(dist)
            
            score = 0
            for idx, i in enumerate(all_dist):
                dist_threshold = all_scale[-1] * all_scale[idx] * 100
                if i > dist_threshold:
                    score += 1

            if score == len(all_dist):
                break
            
        if limit_reached:
            all_scale.pop()
            continue
        
        all_img_cls.append(random.choice(selected_classes))
        all_img_file.append(random.choice([f for f in os.listdir(os.path.join(CLS_PATH, all_img_cls[-1])) if not f.startswith('.')]))

        img = cv2.imread(os.path.join(CLS_PATH, all_img_cls[-1],all_img_file[-1]))
        img = cv2.resize(img, (int(100*scale), int(100*scale)))
        img = np.stack(tuple([np.pad(img[:,:,c], padding_sides, pad_with, padder=255) for c in range(3)]), 2)

        location.append((tx, ty))

        T = np.float32([[1,0,location[-1][0]],[0,1,location[-1][1]]])
        angle = random.randint(0,0)

        wh = img.shape[:2]

        R = cv2.getRotationMatrix2D((wh[0]/2, wh[1]/2), angle, 1)

        img = cv2.warpAffine(img,T,wh, borderValue=(255,255,255))
        img = cv2.warpAffine(img, R, wh, borderValue=(255,255,255)) 
        
        all_img.append(img)

        masks.append(get_mask(img))
        bb1, bb2 = get_bounding_box(img, masks[-1])

        # bb1 = [int(i) for i in bb1]
        # bb2 = [int(i) for i in bb2]


        # img = cv2.rectangle(img, (bb1[0],bb1[1]), (bb1[2], bb1[3]),(0,255,0),2)
        # img = cv2.rectangle(img, (bb2[0],bb2[1]), (bb2[2], bb2[3]),(255,0,0),2)

        # cv2.imshow("",img)
        # cv2.waitKey(0)

        bboxes.append(bb2)

    img = all_img[0]
    for i in range(len(all_img)-1):
        mask_overlay = (all_img[i+1] < 250)

        img[mask_overlay] = all_img[i+1][mask_overlay]

    label_dict = generate_label(name, id, masks, bboxes, all_img_cls)

    img = apply_texture(img, masks)

    cv2.imwrite("/Users/georgesnomicos/fridge_net/fruit_dataset2/" + name, img)
    with open("/Users/georgesnomicos/fridge_net/fruit_dataset2/" + label_name, "w+") as f:
        json.dump(label_dict, f)

    return img, label_dict

def generate_label(name, id, masks, bboxes, all_img_cls):
    dataset_dicts = []

    # img = cv2.imread("./10_100.jpg")
    # img = cv2.rectangle(img, (bbox[0],bbox[1]), (bbox[2], bbox[3]), (255,0,0))
    # cv2_imshow(img)

    record = {}
    objs = []

    record['file_name'] = "/Users/georgesnomicos/fridge_net/fruit_dataset2/" + name
    record['image_id'] = id
    record["height"] = 256
    record["width"] = 256

    i=0
    for mask, bbox in zip(masks, bboxes):
        coco_rle = str(encode(np.asarray(mask,dtype=np.uint8, order="F")))
        obj = {
            "bbox": list(bbox),
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": coco_rle,
            "category_id": class_idx[all_img_cls[i]],
            "iscrowd": 0
        }
        objs.append(obj)
        i+=1

    record['annotations'] = objs

    return record

if __name__ == '__main__':
    for i in range(10000):
        _, _ = generate_sample(i)
        # cv2.imshow('im',img)
        # cv2.waitKey(0)
