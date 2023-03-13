from modules.DetectronNetwork import DetectronNetwork
from modules.Incubator import Incubator
from modules.generational_image import GenerationImage

if __name__ == "__main__":
    image1 = "input_images\image1.jpg"
    image2 = "input_images\image2.jpg"
    detectron = DetectronNetwork()
    parent1 = GenerationImage(imagepath=image1, generation=0, detectron=detectron)
    parent2 = GenerationImage(imagepath=image2, generation=0, detectron=detectron)
    incubator = Incubator(parent1, parent2).breed(4).save()
    breakpoint()
