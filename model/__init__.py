import warnings
import torch
from torch import optim

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
        self.detector.eval()
        return self.detector(images)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=self.lr or self.learning_rate, weight_decay=self.weight_decay,
                              momentum=self.momentum)

        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=3, min_lr=0)

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler, 'monitor': 'val_accuracy'}

    def training_step(self, batch, batch_idx):
        images, boxes, labels = batch
        targets = self._convert_gt_annotations_to_targets(boxes, labels)

        # fasterrcnn takes both images and targets for training
        loss_dict = self.detector(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", loss, batch_size=self.batch_size, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        return self._evaluate(batch, "val_accuracy", self.val_mean_precision_recall)

    def validation_epoch_end(self, validation_step_outputs):
        val_mean_precision_recall = {"val_" + k: v for k, v in self.val_mean_precision_recall.compute().items()}
        # val_mean_precision_per_class = val_mean_precision_recall.pop("val_map_per_class")
        # val_mean_recall_per_class = val_mean_precision_recall.pop("val_mar_100_per_class")

        self.log_dict(val_mean_precision_recall, sync_dist=True)
        # self.log_dict({f"val_map_": val_mean_precision_per_class}, sync_dist=True)
        # self.log_dict({f"val_mar_100_": val_mean_recall_per_class}, sync_dist=True)

        self.val_mean_precision_recall.reset()

    def test_step(self, batch, batch_idx):
        return self._evaluate(batch, "test_accuracy", self.test_mean_precision_recall)

    def test_epoch_end(self, test_step_outputs):
        test_mean_precision_recall = {"test_" + k: v for k, v in self.test_mean_precision_recall.compute().items()}

        self.log_dict(test_mean_precision_recall, sync_dist=True)

        self.test_mean_precision_recall.reset()

    def _evaluate(self, batch, accuracy_var_name, mean_precision_recall):
        images, boxes, labels = batch
        pred_boxes = self.forward(images)

        accuracy = torch.mean(
            torch.stack([self._accuracy(b, pb["boxes"], 0.1) for b, pb in zip(boxes, pred_boxes)]))
        self.log(accuracy_var_name, accuracy, batch_size=self.batch_size, on_step=False, on_epoch=True)

        targets = self._convert_gt_annotations_to_targets(boxes, labels)

        mean_precision_recall.update(pred_boxes, targets)

        return accuracy

    @staticmethod
    def _convert_gt_annotations_to_targets(boxes, labels):
        targets = []
        for b, l in zip(boxes, labels):
            targets.append({
                "boxes": b,
                "labels": l
            })
        return targets

    @staticmethod
    def _accuracy(src_boxes, pred_boxes, iou_threshold=1.):
        total_gt = len(src_boxes)
        total_pred = len(pred_boxes)
        if total_gt > 0 and total_pred > 0:
            matcher = Matcher(iou_threshold, iou_threshold, allow_low_quality_matches=False)
            match_quality_matrix = box_iou(src_boxes, pred_boxes)

            results = matcher(match_quality_matrix)

            true_positive = torch.count_nonzero(results.unique() != -1)
            matched_elements = results[results > -1]

            false_positive = torch.count_nonzero(results == -1) + (
                    len(matched_elements) - len(matched_elements.unique()))
            false_negative = total_gt - true_positive

            return true_positive / (true_positive + false_positive + false_negative)

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
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    backbone = resnet_fpn_backbone(resnet_name, pretrained, trainable_layers=trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes, box_nms_thresh=box_nms_threshold, max_size=max_image_size, **kwargs)
    if pretrained:
        overwrite_eps(model, 0.0)

    return model
