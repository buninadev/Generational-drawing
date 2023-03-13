import numpy as np, cv2
from modules.generational_image import GenerationImage


class Incubator:
    def __init__(
        self,
        parent1: GenerationImage,
        parent2: GenerationImage,
    ) -> None:
        self.parent1 = parent1
        self.parent2 = parent2
        self.generations = [[parent1, parent2]]
        self.detectron = parent1.detectron

    def create_next_generation(self) -> None:
        if len(self.generations) == 0:
            raise Exception("No generations have been created yet")
        generation = []
        parents = self.generations[-1]
        # mutate with other parent in parents once
        for i, parent in enumerate(parents):
            for other_parent in parents[i + 1 :]:
                children = self.create_children(parent, other_parent)
                for child in children:
                    child_image = self.construct_image(child)
                    generation.append(
                        GenerationImage(
                            generation=len(self.generations),
                            detectron=self.detectron,
                            image_array=child_image,
                        )
                    )
        self.generations.append(generation)
        return generation

    def create_children(
        self, parent1: GenerationImage, parent2: GenerationImage
    ) -> list:
        child1Layers = [parent1.back_ground]
        child2Layers = [parent2.back_ground]
        child3Layers = [self.randomly_select(parent1.back_ground, parent2.back_ground)]
        children = [child1Layers, child2Layers, child3Layers]
        for child in children:
            important_objects_from_parents = [
                *parent1.important_objects,
                *parent2.important_objects,
            ]
            number_of_important_objects_to_select = np.random.randint(
                np.floor(len(important_objects_from_parents) / 2),
                len(important_objects_from_parents),
            )
            for _ in range(number_of_important_objects_to_select):
                random_object_index = np.random.randint(
                    0, len(important_objects_from_parents)
                )
                random_object = important_objects_from_parents[random_object_index]
                # remove the object from the list so it can't be selected again
                important_objects_from_parents.pop(random_object_index)
                child.append(random_object)
        return children

    def construct_image(self, layers: list) -> np.ndarray:
        if len(layers) < 1:
            raise ValueError("Too few layers")
        image = layers[0]
        for layer in layers[1:]:
            # paste layer on top of image
            image = self.add_obj(image, layer)
        return image

    def breed(self, generations: int) -> None:
        for _ in range(generations):
            self.create_next_generation()

    def last_generation(self) -> list:
        return self.generations[-1]

    def get_generation(self, generation_number: int) -> list[GenerationImage]:
        return self.generations[generation_number]

    # randmly select pixels from two images
    def randomly_select(self, image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
        # get max width and height
        width = max(len(image1), len(image2))
        height = max(len(image1[0]), len(image2[0]))
        image_to_construct = np.zeros((width, height, 3), np.uint8)
        for i in range(len(image_to_construct)):
            for j in range(len(image_to_construct[0])):
                if np.random.random() < 0.5:
                    if i < len(image1) and j < len(image1[0]):
                        image_to_construct[i][j] = image1[i][j]
                    else:
                        image_to_construct[i][j] = image2[i][j]
                else:
                    if i < len(image2) and j < len(image2[0]):
                        image_to_construct[i][j] = image2[i][j]
                    else:
                        image_to_construct[i][j] = image1[i][j]
        return image_to_construct

    def save(self, generation_number: int = -1) -> None:
        if generation_number == -1:
            generation = self.last_generation()
        else:
            generation = self.get_generation(generation_number)
        for image in generation:
            image.save()

    def add_obj(self, background, img, x=None, y=None):
        """
        Arguments:
        background - background image in CV2 RGB format
        img - image of object in CV2 RGB format
        x, y - coordinates of the center of the object image
        0 < x < width of background
        0 < y < height of background

        Function returns background with added object in CV2 RGB format

        CV2 RGB format is a numpy array with dimensions width x height x 3

        """
        bg = background.copy()
        if x is None:
            x = np.random.randint(0, bg.shape[1])
        if y is None:
            y = np.random.randint(0, bg.shape[0])

        # get width and height of the object image
        h, w = img.shape[:2]

        # calculate the top and bottom of the ROI
        top, bottom = y - h // 2, y + h // 2
        # calculate the left and right of the ROI
        left, right = x - w // 2, x + w // 2

        # check if the ROI is outside the boundaries of the background
        if top < 0:
            h += top
            top = 0
        if bottom > bg.shape[0]:
            h -= bottom - bg.shape[0]
            bottom = bg.shape[0]
        if left < 0:
            w += left
            left = 0
        if right > bg.shape[1]:
            w -= right - bg.shape[1]
            right = bg.shape[1]

        # add the object to the background
        bg[top:bottom, left:right] += img[:h, :w]

        return bg
