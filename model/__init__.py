import warnings
import torch

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection._utils import overwrite_eps, Matcher

from pytorch_lightning import LightningModule

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import box_iou


class MC2CNN(LightningModule):
    def __init__(self, resnet_name, n_classes=2, lr_rate=1e-4, batch_size=4, weight_decay=0.5e-4, momentum=0.9,
                 box_nms_threshold=0.5, max_image_size=1333):
        super().__init__()

        assert resnet_name in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")

        self.train_loss = None
        self.val_loss = None
        self.test_loss = None

        self.val_accuracy = None
        self.test_accuracy = None

        self.val_mean_precision_recall = MeanAveragePrecision(class_metrics=True)
        self.test_mean_precision_recall = MeanAveragePrecision(class_metrics=True)

        self.detector = _fasterrcnn_resnet_fpn(resnet_name=resnet_name, pretrained=True,
                                               box_nms_threshold=box_nms_threshold, max_image_size=max_image_size)

        in_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)

        self.lr = lr_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.momentum = momentum

    def forward(self, images, targets=None):
        # Torchvision FasterRCNN returns the loss during training
        # and the boxes during eval
        self.detector.eval()
        return self.detector(images)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr or self.learning_rate,
                                    weight_decay=self.weight_decay, momentum=self.momentum)

        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=3,
                                                                  min_lr=0)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'val_accuracy'}

    def training_step(self, batch, batch_idx):
        images, boxes, labels = batch
        targets = []

        for b, l in zip(boxes, labels):
            targets.append({
                "boxes": b,
                "labels": l
            })

        # fasterrcnn takes both images and targets for training
        loss_dict = self.detector(images, targets)
        loss = sum(loss for loss in loss_dict.values())

        self.log("train_loss", loss, batch_size=self.batch_size, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, boxes, labels = batch
        pred_boxes = self.forward(images)

        self.val_accuracy = torch.mean(
            torch.stack([self._accuracy(b, pb["boxes"], 0.1) for b, pb in zip(boxes, pred_boxes)]))
        self.log("val_accuracy", self.val_accuracy, batch_size=self.batch_size, on_step=False, on_epoch=True)

        targets = []
        for b, l in zip(boxes, labels):
            targets.append({
                "boxes": b,
                "labels": l
            })

        self.val_mean_precision_recall.update(pred_boxes, targets)

        return self.val_accuracy

    def validation_epoch_end(self, validation_epoch_outputs):
        val_mean_precision_recall = {"val_" + k: v for k, v in self.val_mean_precision_recall.compute().items()}

        # val_mean_precision_per_class = val_mean_precision_recall.pop("val_map_per_class")
        # val_mean_recall_per_class = val_mean_precision_recall.pop("val_mar_100_per_class")

        self.log_dict(val_mean_precision_recall, sync_dist=True)

        # self.log_dict({f"val_map_": val_mean_precision_per_class}, sync_dist=True)
        #
        # self.log_dict({f"val_mar_100_": val_mean_recall_per_class}, sync_dist=True)

        self.val_mean_precision_recall.reset()

    def test_step(self, batch, batch_idx):
        images, boxes, labels = batch
        pred_boxes = self.forward(images)

        self.test_accuracy = torch.mean(
            torch.stack([self._accuracy(b, pb["boxes"], 0.1) for b, pb in zip(boxes, pred_boxes)]))
        self.log("test_accuracy", self.test_accuracy, batch_size=self.batch_size, on_step=False, on_epoch=True)

        targets = []
        for b, l in zip(boxes, labels):
            targets.append({
                "boxes": b,
                "labels": l
            })

        self.test_mean_precision_recall.update(pred_boxes, targets)

        return self.test_accuracy

    def test_epoch_end(self, test_epoch_outputs):
        test_mean_precision_recall = {"test_" + k: v for k, v in self.test_mean_precision_recall.compute().items()}

        # test_mean_precision_per_class = test_mean_precision_recall.pop("test_map_per_class")
        # test_mean_recall_per_class = test_mean_precision_recall.pop("test_mar_100_per_class")

        self.log_dict(test_mean_precision_recall, sync_dist=True)

        # self.log_dict({f"test_map_": value for value in test_mean_precision_per_class}, sync_dist=True)
        #
        # self.log_dict({f"test_mar_100_": value for value in test_mean_recall_per_class}, sync_dist=True)

        self.test_mean_precision_recall.reset()

    @staticmethod
    def _accuracy(src_boxes, pred_boxes, iou_threshold=1.):
        """
        The accuracy method is not the one used in the evaluator but very similar
        """
        total_gt = len(src_boxes)
        total_pred = len(pred_boxes)
        if total_gt > 0 and total_pred > 0:

            # Define the matcher and distance matrix based on iou
            matcher = Matcher(iou_threshold, iou_threshold,
                              allow_low_quality_matches=False)
            match_quality_matrix = box_iou(src_boxes, pred_boxes)

            results = matcher(match_quality_matrix)

            true_positive = torch.count_nonzero(results.unique() != -1)
            matched_elements = results[results > -1]

            # in Matcher, a pred element can be matched only twice
            false_positive = torch.count_nonzero(
                results == -1) + (len(matched_elements) - len(matched_elements.unique()))
            false_negative = total_gt - true_positive

            accuracy = true_positive / (true_positive + false_positive + false_negative)
            # precision = true_positive / (true_positive + false_positive)
            # recall = true_positive / (true_positive + false_negative)
            # f1_score = (precision * recall) / ((precision + recall) / 2)
            return accuracy

        elif total_gt == 0:
            if total_pred > 0:
                return torch.tensor(0.).cuda()
            else:
                return torch.tensor(1.).cuda()
        elif total_gt > 0 and total_pred == 0:
            return torch.tensor(0.).cuda()


def _validate_trainable_layers(pretrained, trainable_backbone_layers, max_value, default_value):
    # don't freeze any layers if pretrained model or backbone is not used
    if not pretrained:
        if trainable_backbone_layers is not None:
            warnings.warn(
                "Changing trainable_backbone_layers has not effect if "
                "neither pretrained nor pretrained_backbone have been set to True, "
                "falling back to trainable_backbone_layers={} so that all layers are trainable".format(max_value))
        trainable_backbone_layers = max_value

    # by default freeze first blocks
    if trainable_backbone_layers is None:
        trainable_backbone_layers = default_value
    assert 0 <= trainable_backbone_layers <= max_value
    return trainable_backbone_layers


def _fasterrcnn_resnet_fpn(resnet_name, pretrained=False, num_classes=91, pretrained_backbone=True,
                           trainable_backbone_layers=None, box_nms_threshold=0.5, max_image_size=1333, **kwargs):
    """
    Constructs a Faster R-CNN model with a ResNet-101-FPN backbone.

    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:

        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box

    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:

        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction

    Faster R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.

    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
            :param trainable_backbone_layers:
            :param pretrained_backbone:
            :param num_classes:
            :param pretrained:
            :param resnet_name:
    """
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    backbone = resnet_fpn_backbone(resnet_name, pretrained, trainable_layers=trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes, box_nms_thresh=box_nms_threshold, max_size=max_image_size, **kwargs)
    if pretrained:
        overwrite_eps(model, 0.0)

    return model
