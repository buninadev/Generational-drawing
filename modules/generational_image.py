from os import makedirs
import cv2
import numpy as np
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer

from modules.DetectronNetwork import DetectronNetwork


class GenerationImage:
    def __init__(
        self,
        imagepath: str = None,
        generation: int = 0,
        detectron: DetectronNetwork = None,
        conf: float = 0.5,
        image_array: np.ndarray = None,
        image_name: str = None,
    ):
        if imagepath == None:
            self.img0 = image_array
            self.image = self.img0.copy()
        else:
            try:
                # check if the image exists
                open(imagepath)

            except Exception as e:
                raise FileNotFoundError(
                    "Image not found, please check input_images folder"
                )
            self.img0 = cv2.imread(imagepath)
            self.image = self.img0.copy()
        self.generation = generation
        self.detected_objects = []
        default_name = imagepath.split("\\")[-1] if imagepath else "gen_image"
        self.image_name = image_name if image_name else default_name
        self.outputpath = (
            "output_images\gen_" + str(self.generation) + "_" + self.image_name
        )
        self.conf = conf
        self.detectron = detectron
        self.outputs = self.detect()
        self.overlayed_image = self.get_mask_from_outputs()
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

    def get_background(self) -> np.ndarray:
        background = self.image.copy()
        antimask = np.ones(self.image.shape, self.image.dtype)
        for mask in self.outputs["instances"].to("cpu").pred_masks:
            antimask[mask] = 0
        return cv2.bitwise_and(background, background, mask=antimask[:, :, 2])

    # save the image
    def save(self, folder: str = None, overlay: bool = False):
        if folder:
            makedirs(folder, exist_ok=True)
            if overlay:
                cv2.imwrite(
                    folder + "\gen_" + str(self.generation) + "_" + self.image_name,
                    self.overlayed_image,
                )
            else:
                cv2.imwrite(
                    folder + "\gen_" + str(self.generation) + "_" + self.image_name,
                    self.image,
                )
        else:
            if overlay:
                cv2.imwrite(self.outputpath, self.overlayed_image)
            else:
                cv2.imwrite(self.outputpath, self.image)

    # clear class from memory
    def clear(self):
        self.important_objects = None
        self.masks = None
        self.outputs = None
        self.overlayed_image = None


# Path: modules\generational_image.py
