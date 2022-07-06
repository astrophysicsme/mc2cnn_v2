import numpy as np

from PIL import Image

from torchvision.transforms import functional as f

from transformations.augmentations import ImgAugTransform

import albumentations
from albumentations.pytorch import ToTensorV2


def transform(stage="test"):
    assert stage in ("train", "val", "test")

    transformations = [TopPadding(), ToTensor()]
    # TODO: implement ImgAug transformations
    # if stage == "train":
    #     transformations_array.append(ImgAugTransform())

    return Compose(transformations)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class TopPadding(object):
    """
    Transformation to add top padding to an image to normalize astro compressed images
    """

    def __call__(self, image, target):
        if image.size == (1920, 608):
            # transform image
            # convert image to array
            image_data = np.asarray(image)
            # create empty image array
            white_image_data = np.full(shape=(1216, 1920, 3), fill_value=(250, 250, 250), dtype=np.uint8)
            # copy image array to the empty image array
            white_image_data[608:1216, 0:1920] = image_data
            # convert the modified white image array to a PIL Image
            image = Image.fromarray(white_image_data)

            # transform target
            for t in range(0, len(target)):
                target[t, 1] += 608
                target[t, 3] += 608

        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = f.to_tensor(image)
        return image, target
