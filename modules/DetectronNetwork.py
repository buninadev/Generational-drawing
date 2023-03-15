# Setup detectron2 logger
# from detectron2.utils.logger import setup_logger

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import torch


class DetectronNetwork:
    def __init__(self):
        self.cfg = get_cfg()
        self.model_path = "COCO-PanopticSegmentation/panoptic_fpn_R_50_3x.yaml"
        self.cfg.merge_from_file(model_zoo.get_config_file(self.model_path))
        self.cfg.MODEL.DEVICE = "cpu"
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self.model_path)
        self.predictor = DefaultPredictor(self.cfg)

    def get_outputs(self, image):
        with torch.no_grad():
            outputs = self.predictor(image)
        return outputs
