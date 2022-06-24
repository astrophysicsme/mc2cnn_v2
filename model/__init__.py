import warnings
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional, Literal, Union, Any

import torch
from torch import optim

from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.models.detection._utils import overwrite_eps, Matcher

from pytorch_lightning import LightningModule

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.ops import box_iou

from metrics import PalletLevelPrecisionRecall


class MC2CNN(LightningModule):
    def __init__(
            self,
            resnet_name: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
            n_classes: Optional[int] = 2,
            lr_rate: Optional[float] = 1e-4,
            batch_size: Optional[int] = 4,
            weight_decay: Optional[float] = 0.5e-4,
            momentum: Optional[float] = 0.9,
            box_nms_threshold: Optional[float] = 0.5,
            max_image_size: Optional[int] = 1333,
            pallet_manipulations: Optional[int] = 4,
            pallet_manipulations_identifiers: Optional[tuple] = ("", "hw", "vc", "vchw"),
            views_per_pallet: Optional[int] = 30,
            passes_per_pallet: Optional[int] = 5,
            views_per_pass: Optional[int] = 6,
    ):
        super().__init__()

        assert resnet_name in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")

        self.train_loss = None
        self.val_loss = None
        self.test_loss = None

        self.train_accuracy = None
        self.val_accuracy = None
        self.test_accuracy = None

        self.val_mean_precision_recall = MeanAveragePrecision(class_metrics=True)
        self.test_mean_precision_recall = MeanAveragePrecision(class_metrics=True)

        self.test_pallet_level_precision_recall = PalletLevelPrecisionRecall(
            file_name="test_truth_table",
            passes_per_pallet=passes_per_pallet,
            views_per_pass=views_per_pass,
            pallet_manipulations=pallet_manipulations,
            pallet_manipulations_identifiers=pallet_manipulations_identifiers,
            views_per_pallet=views_per_pallet,
        )

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
        accuracy, loss = self._evaluate(batch, False, prefix="train")
        return loss

    def validation_step(self, batch, batch_idx):
        accuracy, loss = self._evaluate(batch, self.val_mean_precision_recall, prefix="val")
        self.log("val_accuracy", accuracy, batch_size=self.batch_size, on_step=False, on_epoch=True)
        return {"val_accuracy": accuracy, "val_loss": loss}

    def validation_epoch_end(self, validation_step_outputs):
        val_mean_precision_recall = {"val_" + k: v for k, v in self.val_mean_precision_recall.compute().items()}
        self.log_dict(val_mean_precision_recall, sync_dist=True)
        self.val_mean_precision_recall.reset()

    def test_step(self, batch, batch_idx):
        accuracy, loss = self._evaluate(batch, self.test_mean_precision_recall,
                                        pallet_level_precision_recall=self.test_pallet_level_precision_recall,
                                        prefix="test")
        return {"test_accuracy": accuracy, "test_loss": loss}

    def test_epoch_end(self, test_step_outputs):
        test_mean_precision_recall = {"test_" + k: v for k, v in self.test_mean_precision_recall.compute().items()}
        self.log_dict(self.test_pallet_level_precision_recall.compute(), sync_dist=True)
        self.log_dict(test_mean_precision_recall, sync_dist=True)
        self.test_mean_precision_recall.reset()

    def _evaluate(
            self,
            batch,
            mean_precision_recall: Union[MeanAveragePrecision, bool],
            pallet_level_precision_recall: Optional[Union[PalletLevelPrecisionRecall, bool]] = False,
            prefix: Literal["train", "val", "test"] = "val"
    ):
        images, boxes, labels, image_file_names = batch
        targets = self._convert_gt_annotations_to_targets(boxes, labels)

        loss_dict, pred_boxes = self._eval_forward(self.detector, images, targets)

        accuracy = torch.mean(
            torch.stack([self._accuracy(b, pb["boxes"], 0.1) for b, pb in zip(boxes, pred_boxes)]))
        self.logger.experiment.add_scalars('accuracy', {f"{prefix}": accuracy}, self.global_step)

        loss = sum(loss for loss in loss_dict.values())
        self.logger.experiment.add_scalars('loss', {f"{prefix}": loss}, self.global_step)

        minimized_pred_boxes = self._minimize_predicted_boxes(pred_boxes)

        if pallet_level_precision_recall:
            pallet_level_precision_recall.update(image_file_names, targets, minimized_pred_boxes)

        if mean_precision_recall:
            mean_precision_recall.update(minimized_pred_boxes, targets)

        return accuracy, loss

    def _minimize_predicted_boxes(self, pred_boxes, score_threshold: float = 0.1):
        minimized_pred_boxes = list()

        for p in range(len(pred_boxes)):
            bs = list()
            ls = list()
            ss = list()

            b_l = pred_boxes[p]["boxes"].tolist()
            l_l = pred_boxes[p]["labels"].tolist()
            s_l = pred_boxes[p]["scores"].tolist()
            for s in range(len(s_l)):
                if s_l[s] > score_threshold:
                    bs.append(b_l[s])
                    ls.append(l_l[s])
                    ss.append(s_l[s])

            bs = torch.tensor(bs, dtype=torch.float32, device=self.device)
            ls = torch.tensor(ls, dtype=torch.int64, device=self.device)
            ss = torch.tensor(ss, dtype=torch.float32, device=self.device)

            minimized_pred_boxes.append({
                "boxes": bs,
                "labels": ls,
                "scores": ss
            })

        return minimized_pred_boxes

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
    def _accuracy(src_boxes, pred_boxes, iou_threshold: Optional[float] = 1.):
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

    @staticmethod
    def _eval_forward(model, images, targets):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                It returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
                :param targets:
                :param images:
                :param model:
        """
        model.eval()

        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = model.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

        features = model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        model.rpn.training = True
        # model.roi_heads.training=True

        # proposals, proposal_losses = model.rpn(images, features, targets)
        features_rpn = list(features.values())
        objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
        anchors = model.rpn.anchor_generator(images, features_rpn)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        proposals, scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        assert targets is not None
        labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
        regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
        loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
            objectness, pred_bbox_deltas, labels, regression_targets
        )
        proposal_losses = {
            "loss_objectness": loss_objectness,
            "loss_rpn_box_reg": loss_rpn_box_reg,
        }

        # detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
        image_shapes = images.image_sizes
        proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals,
                                                                                                      targets)
        box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
        box_features = model.roi_heads.box_head(box_features)
        class_logits, box_regression = model.roi_heads.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []

        loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
        detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals,
                                                                       image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )
        detections = result
        detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        model.rpn.training = False
        model.roi_heads.training = False
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses, detections


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


def _fasterrcnn_resnet_fpn(
        resnet_name: Literal["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
        pretrained: Optional[bool] = False,
        num_classes: Optional[int] = 91,
        pretrained_backbone: Optional[bool] = True,
        trainable_backbone_layers=None,
        box_nms_threshold: Optional[float] = 0.5,
        max_image_size: Optional[int] = 1333,
        **kwargs: Optional[Any]
):
    assert resnet_name in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3)

    backbone = resnet_fpn_backbone(resnet_name, pretrained, trainable_layers=trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes, box_nms_thresh=box_nms_threshold, max_size=max_image_size, **kwargs)
    if pretrained:
        overwrite_eps(model, 0.0)

    return model
