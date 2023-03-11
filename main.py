from modules.DetectronNetwork import DetectronNetwork
from modules.generational_image import GenerationImage

if __name__ == "__main__":
    image1 = "input_images\image1.jpg"
    image2 = "input_images\image2.jpg"
    detectron = DetectronNetwork()
    parent1 = GenerationImage(image1, 0, detectron).save()
    parent2 = GenerationImage(image2, 0, detectron).save()
