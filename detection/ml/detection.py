import os
import sys
import random
import math
import numpy as np
import skimage.io
import colorsys
import cv2
from django.conf import settings

ROOT_DIR = settings.BASE_DIR + "/detection/ml/"
# ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)  # To find local version of the library
# Import Mask RCNN
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

# Import COCO config
sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))  # To find local version
from samples.coco import coco

MODEL_DIR = os.path.join(ROOT_DIR, "logs")


class ModelConfig:

    def __init__(self, model_path="mask_rcnn_coco.h5"):

        # Directory to save logs and trained model

        # Local path to trained weights file
        self.COCO_MODEL_PATH = os.path.join(settings.MEDIA_ROOT, 'models', model_path)
        # COCO_MODEL_PATH = "/content/drive/My Drive/Tennis_ball_with_m-rcnn/logs/mask_rcnn_tennisball.h5"


        # Download COCO trained weights from Releases if needed
        if not os.path.exists(self.COCO_MODEL_PATH):
            utils.download_trained_weights(self.COCO_MODEL_PATH)

        # Directory of images to run detection on
        # IMAGE_DIR = os.path.join(ROOT_DIR, "images")

        class InferenceConfig(coco.CocoConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()

        # Create model object in inference mode.
        self.model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        self.model.keras_model._make_predict_function()

        # Load weights trained on MS-COCO
        self.model.load_weights(self.COCO_MODEL_PATH, by_name=True)
        # model.keras_model._make_predict_function()

        # COCO Class names
        # Index of the class in the list is its ID. For example, to get ID of
        # the teddy bear class, use: class_names.index('teddy bear')
        # class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
        #                'bus', 'train', 'truck', 'boat', 'traffic light',
        #                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
        #                'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
        #                'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        #                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        #                'kite', 'baseball bat', 'baseball glove', 'skateboard',
        #                'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        #                'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        #                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        #                'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
        #                'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        #                'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
        #                'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
        #                'teddy bear', 'hair drier', 'toothbrush']

    def detect_object(self, image_url, classes, image_dir="/ml/image/temp.jpg"):
        # file_names = next(os.walk(IMAGE_DIR))[2]
        # image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))

        # MEDIA_ROOT = "E:\Temp\object_detection\media"
        image = skimage.io.imread(image_url)

        # Run detection
        results = self.model.detect([image], verbose=0)
        # Visualize results
        r = results[0]

        CLASS_NAMES = classes
        hsv = [(i / len(CLASS_NAMES), 1, 1.0) for i in range(len(CLASS_NAMES))]
        COLORS = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))

        for i in range(0, r["rois"].shape[0]):
            # extract the class ID and mask for the current detection, then
            # grab the color to visualize the mask (in BGR format)
            classID = r["class_ids"][i]
            mask = r["masks"][:, :, i]
            color = COLORS[classID][::-1]

            # visualize the pixel-wise mask of the object
            image = visualize.apply_mask(image, mask, color, alpha=0.5)

        # convert the image back to BGR so we can use OpenCV's drawing
        # functions
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # loop over the predicted scores and class labels
        # for i in range(0, len(r["scores"])):
        #     # extract the bounding box information, class ID, label, predicted
        #     # probability, and visualization color
        #     (startY, startX, endY, endX) = r["rois"][i]
        #     classID = r["class_ids"][i]
        #     label = CLASS_NAMES[classID]
        #     score = r["scores"][i]
        #     color = [int(c) for c in np.array(COLORS[classID]) * 255]

            # draw the bounding box, class label, and score of the object
            # cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
            # text = "{}: {:.3f}".format(label, score)
            # y = startY - 10 if startY - 10 > 10 else startY + 10
            # cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.6, color, 2)

        # show the output image
        cv2.imwrite(settings.STATIC_DIR + "/ml/temp.jpg", image)
        # cv2.imshow("Output", image)
        # cv2.waitKey()
        return r