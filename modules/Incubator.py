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
        self.total_number_of_children = 0

    def create_next_generation(self) -> None:
        if len(self.generations) == 0:
            raise Exception("No generations have been created yet")
        generation = []
        if len(self.generations) > 2:
            for gen in self.generations[-2]:
                gen.clear()
        parents = self.generations[-1]
        generation_to_create = parents[0].generation + 1
        print(f"Creating generation {generation_to_create}")
        # mutate with other parent in parents once
        child_id = 0
        for i, parent in enumerate(parents):
            if i == len(parents) - 1:
                break
            if len(parents) >= 6 and bool(np.random.randint(0, 2)):
                continue
            other_parent = parents[i + 1]
            children = self.create_children(parent, other_parent)
            for child in children:
                child_image = self.construct_image(child)
                generation.append(
                    GenerationImage(
                        generation=generation_to_create,
                        detectron=self.detectron,
                        image_array=child_image,
                        image_name=f"child{child_id}.jpg",
                    )
                )
                child_id += 1
        self.generations.append(generation)
        print(f"{len(self.generations[-1])} children created")
        return self

    def create_children(
        self, parent1: GenerationImage, parent2: GenerationImage
    ) -> list:
        child1Layers = [parent1.back_ground]
        child2Layers = [parent2.back_ground]
        child3Layers = [self.randomly_select(parent1.back_ground, parent2.back_ground)]
        children = [child1Layers, child2Layers, child3Layers]
        for i in range(len(children)):
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
                children[i].append(random_object)
        return children

    def construct_image(self, layers: list) -> np.ndarray:
        if len(layers) < 1:
            raise ValueError("Too few layers")
        image = layers[0]
        for layer in layers[1:]:
            # paste layer on top of image
            image = self.add_obj(image, layer)
        return image

    def breed_incest(self, generations: int):
        for _ in range(generations):
            self.create_next_generation()
        return self

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
        # add image1  to first half
        height_diff = height - len(image1[0])
        image_to_construct[: width // 2, : height - height_diff] += image1[
            : width // 2, : height - height_diff
        ]
        # add image2 to second half
        height_diff = height - len(image1[0])
        image_to_construct[width // 2 + 1 :, : height - height_diff] += image1[
            : width // 2 - 1, : height - height_diff
        ]

        return image_to_construct

    def save(self, generation_number: int = None):
        if generation_number is None:
            for generation in self.generations:
                for image in generation:
                    image.save(folder="incubator_output/gen_" + str(image.generation))
        else:
            generation = self.generations[generation_number]
            for image in generation:
                image.save()
        return self

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
        background_shape = bg[top:bottom, left:right].shape
        img_shape = img[:h, :w].shape
        h_diff = background_shape[0] - img_shape[0]
        w_diff = background_shape[1] - img_shape[1]
        # all colors (except black) become white
        mask = img[: h + h_diff, : w + w_diff] > 0
        # invert mask
        mask = np.logical_not(mask)
        # apply mask to background
        bg[top:bottom, left:right] = bg[top:bottom, left:right] * mask

        # add the object to the background
        bg[top:bottom, left:right] += img[: h + h_diff, : w + w_diff]

        return bg

    # clear class from memory
    def clear(self):
        self.generations = []
        self.detectron = None
        self.back_grounds = []
        self.important_objects = []
        self.objects = []
        self.generation_size = 0
        self.generation_number = 0

    def __del__(self):
        self.clear()
