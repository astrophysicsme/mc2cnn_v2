from os.path import join
from PIL import Image

import torch
from torch.utils.data import Dataset

from pycocotools.coco import COCO


class AstroPallet(Dataset):
    def __init__(self, root, annotation_file_name, transform=None):
        self.root = root
        self.transform = transform
        self.coco = COCO(f"{root}/{annotation_file_name}")
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        # List: get annotation id from coco
        ann_ids = coco.getAnnIds(imgIds=img_id)
        # Dictionary: target coco_annotation file for an image
        coco_annotation = coco.loadAnns(ann_ids)
        # path for input image
        path = coco.loadImgs(img_id)[0]["file_name"]
        # open the input image
        img = Image.open(join(self.root, path)).convert("RGB")
        # number of objects in the image
        num_objs = len(coco_annotation)

        # Bounding boxes for objects
        # In coco format, bbox = [x_min, y_min, width, height]
        # In pytorch, the input should be [x_min, y_min, x_max, y_max]
        boxes = []
        labels = []
        for i in range(num_objs):
            x_min = coco_annotation[i]["bbox"][0]
            y_min = coco_annotation[i]["bbox"][1]
            x_max = x_min + coco_annotation[i]["bbox"][2]
            y_max = y_min + coco_annotation[i]["bbox"][3]
            boxes.append([x_min, y_min, x_max, y_max])

            labels.append(coco_annotation[i]["category_id"])

        boxes = torch.as_tensor(boxes, dtype=torch.long)

        # Labels (In my case, I only one class: target class or background)
        labels = torch.as_tensor(labels, dtype=torch.long)

        # convert img_id to tensor
        # img_id = torch.tensor([img_id])

        # Size of bbox (Rectangular)
        # areas = []
        # for i in range(num_objs):
        #     areas.append(coco_annotation[i]["area"])
        # areas = torch.as_tensor(areas, dtype=torch.long)
        # Iscrowd
        # iscrowd = torch.zeros((num_objs,), dtype=torch.long)

        # if self.transform is not None:
        img, ann = self.transform(img, boxes)

        # Annotation is in dictionary format
        # metadata = {"boxes": ann, "labels": labels, "image_id": img_id, "area": areas, "iscrowd": iscrowd}

        return img, ann, labels

    def __len__(self):
        return len(self.ids)
