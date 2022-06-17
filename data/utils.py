import torch


def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader)

    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """

    images = list()
    targets = list()
    labels = list()

    for i, t, l in batch:
        images.append(i)
        targets.append(t)
        labels.append(l)

    images = torch.stack(images, dim=0)

    return images, targets, labels
