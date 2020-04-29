from detectron2.config import get_cfg
import cv2
from detectron2.utils.visualizer import ColorMode
import json
from mask_rcnn import get_fruit_dicts
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import os
from detectron2.utils.visualizer import Visualizer

with open('class_idx.txt') as f:
    class_idx = json.load(f)

class_idx = {name: i for i, name in enumerate(['Tomato', 'Orange', 'Lemon', 'Apple', 'Banana', 'Avocado'])} 

dataset_dicts = get_fruit_dicts()

MetadataCatalog.get("inference").set(thing_classes=list(class_idx.keys()))
fruit_metadata = MetadataCatalog.get("inference")

cfg = get_cfg()

cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = os.path.join("/Users/georgesnomicos/fridge_net/output", "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set the testing threshold for this model
cfg.MODEL.DEVICE='cpu'
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 1.4]]
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_idx)  

predictor = DefaultPredictor(cfg)

im = cv2.imread("tomato.jpg")
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
                metadata=fruit_metadata, 
                scale=0.8, 
                instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
)
v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("",v.get_image()[:, :, ::-1])
cv2.waitKey(0)


# Decompose network:
im = ImageList(im.unsqueeze(dim=0), [(256,256)])

predictor.model.eval()
# print(predictor.model)
features = predictor.model.backbone(im.tensor)
obj_logits, _ = predictor.model.proposal_generator.rpn_head(features.values())
T = Compose([ToPILImage(), Resize(256), ToTensor()])
softmax = torch.nn.Softmax()
objectness_map = softmax(T(obj_logits[1][0]))


overlap_feature(objectness_map.numpy(), im.tensor[0].numpy())    

proposals, _ = predictor.model.proposal_generator(im, features)
bbox = proposals[0].proposal_boxes

show_bbox(im.tensor[0].numpy(),bbox.tensor[-5:,:].tolist())