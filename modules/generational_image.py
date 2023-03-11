import cv2
import numpy as np


class GenerationImage:
    def __init__(self, imagepath: str, generation: int):
        self.image = cv2.imread(imagepath)
        self.generation = generation
        self.detected_objects = []
        self.image_name = imagepath.split("\\")[-1]
        self.outputpath = (
            "output_images\gen_" + str(self.generation) + "_" + self.image_name
        )
        self.conf = 0.5

    # object detection using YOLO algorithm
    def detect(self, net, outputlayers):
        height, width, channels = self.image.shape
        blob = cv2.dnn.blobFromImage(
            self.image, 0.00392, (416, 416), (0, 0, 0), True, crop=False
        )
        print(blob.shape)
        net.setInput(blob)
        outs = net.forward(outputlayers)
        return np.vstack(outs), height, width

    # get pixel coordinates of the detected object
    def get_coordinates(self, outs, height, width):
        class_ids = []
        confidences = []
        boxes = []
        for output in outs:
            scores = output[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > self.conf:
                x, y, w, h = output[:4] * np.array([width, height, width, height])
                p0 = int(x - w // 2), int(y - h // 2)
                p1 = int(x + w // 2), int(y + h // 2)
                boxes.append([*p0, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(classID)
                # cv.rectangle(img, p0, p1, WHITE, 1)
        return boxes, confidences, class_ids

    def draw(self, boxes, confidences, classIDs, colors, classes):
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf, self.conf - 0.1)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(self.image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(classes[classIDs[i]], confidences[i])
                cv2.putText(
                    self.image,
                    text,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )
                self.detected_objects.append(classes[classIDs[i]])

            return self.detected_objects

    # save the image
    def save(self):
        cv2.imwrite(self.outputpath, self.image)
        return self.outputpath


# Path: modules\generational_image.py
