from cv2 import dnn
import numpy as np


class Yolo:
    def __init__(self):
        self.__name = "Yolo"
        self.__version = "1.0"
        np.random.seed(42)
        self.classes = open("yolo\coco.names").read().strip().split("\n")
        self.colors = np.random.randint(
            0, 255, size=(len(self.classes), 3), dtype="uint8"
        )

    # get the network
    def get_network(
        configpath: str = "yolo\yolov3.cfg", weightpath: str = "yolo\yolov3.weights"
    ):
        net = dnn.readNetFromDarknet(configpath, weightpath)
        net.setPreferableBackend(dnn.DNN_BACKEND_OPENCV)
        return net

    # get the output layers
    def get_outputlayers(net):
        layer_names = net.getLayerNames()
        outputlayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return outputlayers
