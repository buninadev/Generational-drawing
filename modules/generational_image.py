import cv2
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from modules.DetectronNetwork import DetectronNetwork


class GenerationImage:
    def __init__(
        self,
        imagepath: str,
        generation: int,
        detectron: DetectronNetwork,
        conf: float = 0.5,
    ):
        try:
            # check if the image exists
            open(imagepath)
        except Exception as e:
            raise FileNotFoundError("Image not found, please check input_images folder")
        self.img0 = cv2.imread(imagepath)
        self.image = self.img0.copy()
        self.generation = generation
        self.detected_objects = []
        self.image_name = imagepath.split("\\")[-1]
        self.outputpath = (
            "output_images\gen_" + str(self.generation) + "_" + self.image_name
        )
        self.conf = conf
        self.detectron = detectron
        self.outputs = self.detect()
        self.overlayed_image = self.get_mask_from_outputs()
        cv2.imwrite(self.outputpath, self.overlayed_image)
        self.masks = self.get_pred_masks_only()
        self.important_objects = self.get_important_objects()
        self.back_ground = self.get_background()

    # object detection using YOLO algorithm
    def detect(self):
        outs = self.detectron.get_outputs(self.image)
        return outs

    def get_mask_from_outputs(self):
        v = Visualizer(
            self.image[:, :, ::-1],
            MetadataCatalog.get(self.detectron.cfg.DATASETS.TRAIN[0]),
            scale=1.2,
        )
        out = v.draw_instance_predictions(self.outputs["instances"].to("cpu"))
        return out.get_image()[:, :, ::-1]

    def get_pred_masks_only(self):
        masks = []
        for object in self.outputs["instances"].to("cpu").pred_masks:
            mask = np.zeros(self.image.shape, self.image.dtype)
            mask[object] = 255
            masks.append(mask)
        return masks

    def get_important_objects(self):
        important_objects = []
        for mask in self.masks:
            important_objects.append(
                cv2.bitwise_and(self.image, self.image, mask=mask[:, :, 2])
            )
        return important_objects

    def get_background(self):
        background = self.image.copy()
        antimask = np.ones(self.image.shape, self.image.dtype)
        for mask in self.outputs["instances"].to("cpu").pred_masks:
            antimask[mask] = 0
        return cv2.bitwise_and(background, background, mask=antimask[:, :, 2])

    # save the image
    def save(self):
        for i, object in enumerate(self.important_objects):
            cv2.imwrite(
                "output_images\gen_"
                + str(self.generation)
                + "_"
                + str(i)
                + "_"
                + self.image_name,
                object,
            )
        cv2.imwrite(
            "output_images\gen_"
            + str(self.generation)
            + "_background_"
            + self.image_name,
            self.back_ground,
        )
        cv2.imwrite(self.outputpath, self.overlayed_image)


# Path: modules\generational_image.py
