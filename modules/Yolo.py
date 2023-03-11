import numpy as np
from cv2 import dnn


class Yolo:
    def __init__(self):
        self.__name = "Yolo"
        self.__version = "1.0"

    # get the network
    def get_network(version: str = "yolov5"):
        if version == "yolov3":
            configpath: str = "yolo\yolov3.cfg"
            weightpath: str = "yolo\yolov3.weights"
            net = dnn.readNetFromDarknet(configpath, weightpath)
            net.setPreferableBackend(dnn.DNN_BACKEND_OPENCV)
        elif version == "yolov5":
            configpath: str = "yolo\yolov5s.onnx"
            net = dnn.readNetFromONNX(configpath)


        return net

    # get the output layers
    def get_outputlayers(net):
        layer_names = net.getLayerNames()
        outputlayers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
        return outputlayers

    def get_classes():
        return open("yolo\coco.names").read().strip().split("\n")

    def get_colors(classes):
        np.random.seed(42)
        return np.random.randint(0, 255, size=(len(classes), 3), dtype="uint8")
