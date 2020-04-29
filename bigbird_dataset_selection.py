import os 
from PIL import Image
import numpy as np
from pycocotools.mask import encode, toBbox

PATH = "/home/georgesnomicos/bigbird"
SAVE_PATH = "/home/georgesnomicos/bigbird_s_view"

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

step = 15
img_size = 224

for class_ in os.listdir(PATH):
    class_dir = os.path.join(SAVE_PATH,class_)
    if os.path.exists(class_dir):
        for files in os.listdir(class_dir):
            os.remove(os.path.join(class_dir,files))
    else: 
        os.mkdir(class_dir)

    count = 0
    for view in range(1,6,1):
        for n in range(0,358,step):
            name = "NP{}_{}".format(view,n)
            img = os.path.join(PATH, class_, name+".jpg")
            mask = os.path.join(PATH, class_, "masks", name + "_mask.pbm")

            img = Image.open(img)
            mask = Image.open(mask)

            img = np.array(img)
            mask = np.array(mask)

            for row in range(1024):
                if np.max(mask[row,:]) != np.min(mask[row,:]):
                    start_row = row
                    break
            for row in range(1024):
                if np.max(mask[1024 - 1 - row,:]) != np.min(mask[1024 - 1 - row,:]):
                    end_row = 1024 - 1 - row
                    break  

            # obj_h = end_row - start_row
            # margin = int((224 - obj_h) /2)
            # if margin*2 + obj_h != 224:
            #     margin1 = margin+1
            #     margin2 = margin
            # else:
            #     margin1, margin2 = margin, margin
            margin = 25

            h_1  = start_row - margin
            h_2 = end_row + margin

            for col in range(1280):
                if np.max(mask[:,col]) != np.min(mask[:,col]):
                    start_row = col
                    break
            for col in range(1280):
                if np.max(mask[:,1280 - 1 - col]) != np.min(mask[:,1280 - 1 - col]):
                    end_row = 1280 - 1 - col
                    break  

            # obj_h = end_row - start_row
            # margin = int((224 - obj_h) / 2)
            # if margin*2 + obj_h != 224:
            #     margin1 = margin+1
            #     margin2 = margin
            # else:
            #     margin1, margin2 = margin, margin
            w_1  = start_row - margin
            w_2 = end_row + margin

            img = img[h_1:h_2,w_1:w_2,:]

            # img[mask] = 255
            img = Image.fromarray(img)

            # bbox = toBbox(encode(np.asarray(mask,dtype=np.uint8, order="F"))


            img.save(os.path.join(class_dir,"{}.jpg".format(count)))
            
            count += 1


