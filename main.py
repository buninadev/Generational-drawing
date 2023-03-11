from modules.generational_image import GenerationImage
from modules.Yolo import Yolo
import numpy as np


if __name__ == "__main__":
    parent1 = GenerationImage("input_images\gun_man.jpg", 0)
    parent2 = GenerationImage("input_images\woman_flower.jpg", 0)
    net = Yolo.get_network()
    outputlayers = Yolo.get_outputlayers(net)
    outs1, height1, width1 = parent1.detect(net, outputlayers)
    outs2, height2, width2 = parent2.detect(net, outputlayers)
    boxes1, confidences1, class_ids1 = parent1.get_coordinates(outs1, height1, width1)
    boxes2, confidences2, class_ids2 = parent2.get_coordinates(outs2, height2, width2)
    yoyo = Yolo()
    colors = yoyo.colors
    labels = yoyo.classes
    parent1.draw(boxes1, confidences1, class_ids1, colors, labels)
    parent2.draw(boxes2, confidences2, class_ids2, colors, labels)

    parent1.save()
    parent2.save()
