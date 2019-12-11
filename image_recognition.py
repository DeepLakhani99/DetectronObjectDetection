import torch, torchvision
torch.__version__


# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
from skimage import io


# import some common detectron2 utilities
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
'''cap = cv2.VideoCapture("https://statics.sportskeeda.com/wp-content/uploads/2014/09/bell-1411303355.jpg")
if( cap.isOpened() ):
	im=cap.read()
'''
def get_output(image_bytes):
	'''im = io.imread("https://ichef.bbci.co.uk/news/660/cpsprodpb/147CE/production/_106081938_gettyimages-919536292.jpg")
	im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)	'''
	im = image_bytes
	nparr = np.fromstring(im, np.uint8)
	im = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)
	
	cfg = get_cfg()
	cfg.MODEL.DEVICE='cpu'
	cfg.merge_from_file("/home/deep/image_recog/image_recog/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
	cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can either use the https://dl.fbaipublicfiles.... url, or use the following shorthand
	cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
	predictor = DefaultPredictor(cfg)
	outputs = predictor(im)


# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
	outputs["instances"].pred_classes
	outputs["instances"].pred_boxes

# We can use `Visualizer` to draw the predictions on the image.
	v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
	v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

	final = v.get_image()[:, :, ::-1]
	print(final)
	
	return final
#cv2.imshow('output',v.get_image()[:, :, ::-1])
	'''retval,buffer = cv2.imencode('.jpg',v.get_image()[:, :, ::-1])
	response = make_response(buffer.tobytes())
	return response'''










