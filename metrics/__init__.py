import os
import shutil
import time
from ast import literal_eval

import pandas as pd
import numpy as np
from typing import Optional
from torchvision.ops import nms


class PalletLevelPrecisionRecall:
    def __init__(
            self,
            file_name: str,
            log_dir: Optional[str] = "astro_metrics_logs",
            pallet_manipulations_identifiers: Optional[tuple] = ("_vchw.", "_hw.", "_vc.", ""),
            pallet_manipulations: Optional[int] = 4,
            views_per_pallet: Optional[int] = 30,
            passes_per_pallet: Optional[int] = 5,
            views_per_pass: Optional[int] = 6,
            detection_threshold: Optional[float] = 0.75,
    ):
        self._log_dir = log_dir
        self._file_name = file_name
        self._pr_path, self._pr_full_path = self._handle_existing_file(log_dir=self._log_dir, file_name=self._file_name)
        self._pr_columns = [
            "image_file_name",
            "gt_boxes",
            "gt_labels",
            "predicted_boxes",
            "predicted_labels",
            "predicted_scores"
        ]

        self._tt_path, self._tt_full_path = self._handle_existing_file(log_dir=self._log_dir, file_name="truth_table")
        self._tt_columns = [
            "metric name",
            "metric value"
        ]

        self._pallet_manipulations = pallet_manipulations
        self._views_per_pallet = views_per_pallet
        self._passes_per_pallet = passes_per_pallet
        self._views_per_pass = views_per_pass
        self._pallet_manipulations_identifiers = pallet_manipulations_identifiers

        self._detection_threshold = detection_threshold

    def update(
            self,
            image_file_names,
            targets,
            minimized_pred_boxes
    ):
        assert len(image_file_names) == len(targets) == len(minimized_pred_boxes)

        data = []
        for i in range(len(image_file_names)):
            data.append([
                image_file_names[i],
                targets[i]["boxes"].tolist(),
                targets[i]["labels"].tolist(),
                minimized_pred_boxes[i]["boxes"].tolist(),
                minimized_pred_boxes[i]["labels"].tolist(),
                minimized_pred_boxes[i]["scores"].tolist()
            ])

        df = pd.DataFrame(data=data, columns=self._pr_columns)
        if os.path.exists(self._pr_full_path):
            original_df = pd.read_csv(self._pr_full_path)
            df = pd.concat([original_df, df], ignore_index=True)

        df.to_csv(self._pr_full_path, index=False)

    def compute(self):
        if not os.path.exists(self._pr_full_path):
            raise FileNotFoundError
        else:
            assert self._pallet_manipulations == len(self._pallet_manipulations_identifiers)

            pass_level_truth_table = []
            for chunk in pd.read_csv(self._pr_full_path, converters={
                "gt_boxes": literal_eval,
                "gt_labels": literal_eval,
                "predicted_boxes": literal_eval,
                "predicted_labels": literal_eval,
                "predicted_scores": literal_eval,
            }, chunksize=self._pallet_manipulations * self._views_per_pallet):
                # separate all the different manipulations
                pallets_grouped_by_manipulation = list()
                for identifier in self._pallet_manipulations_identifiers:
                    if identifier != "":
                        sub_chunk = chunk["image_file_name"].str.contains(identifier)
                        # order the rows according to the pallets names. pallets are ordered alphabetically not by view
                        pallets_grouped_by_manipulation.append(self._sort_df_by_map(chunk[sub_chunk]))
                        # drop all extracted rows from the main pd chunk
                        chunk = chunk[sub_chunk == False]
                    else:
                        # order the rows according to the pallets names. pallets are ordered alphabetically not by view
                        pallets_grouped_by_manipulation.append(self._sort_df_by_map(chunk))

                # loop over the views from the same pass
                start_position = 0
                for pallet in pallets_grouped_by_manipulation:
                    #   keep the predicted boxes, label and scores, that overlap in 3 consecutive views
                    for i in range(start_position, self._views_per_pallet, self._views_per_pass):
                        next_position = i + self._views_per_pass

                        pass_views = pallet[i:next_position]

                        view_range = f"{i} to {next_position}"
                        file_name, manipulation_prefix = self._extract_file_name_with_manipulation_prefix(
                            pass_views.iloc[0]["image_file_name"])

                        # TODO: add support for the labels and scores when available
                        pred_boxes, pred_labels, pred_scores = self._convert_single_view_to_pass_view_truth_table(
                            pass_views, column_name="predicted_boxes", is_ground_truth=False)
                        gt_boxes, gt_labels = self._convert_single_view_to_pass_view_truth_table(pass_views,
                                                                                                 column_name="gt_boxes",
                                                                                                 is_ground_truth=True)

                        iou_thr = 0.1
                        threats_detected = []

                        pass_level_truth_table.append({
                            "image_file_name": file_name,
                            "manipulation": manipulation_prefix,
                            "views": view_range,
                            "gt_boxes": gt_boxes,
                            "gt_labels": gt_labels,
                            "predicted_boxes": pred_boxes,
                            "predicted_labels": pred_labels,
                            "predicted_scores": pred_scores,
                        })

                #   keep the ground truth boxes and labels, that overlap in 3 consecutive views
                #   calculate iou between the pass predicted boxes and the ground truth boxes
                #       while checking the labels of these boxes
                #   calculate the precision and recall for the pallet with each of its manipulations
                #
                # calculate the precision and recall for the whole dataset

            final_result = {
                "mean avg. precision": 1.0,
                "mean avg. recall": 1.0,
                "mean avg. false alarm rate": 0.0
            }

            final_result_for_pd_df = {
                self._tt_columns[0]: list(final_result.keys()),
                self._tt_columns[1]: list(final_result.values())
            }

            tt_result_df = pd.DataFrame(data=final_result_for_pd_df, columns=self._tt_columns)
            tt_result_df.to_csv(self._tt_full_path, index=False)

            return final_result

    def _convert_single_view_to_pass_view_truth_table(self, pass_views, column_name, is_ground_truth=False):
        pass_boxes = []
        pass_labels = []
        if not is_ground_truth:
            pass_scores = []

        for box, label, score in self._check_three_consecutive_views(pass_views.iloc[0], pass_views.iloc[1],
                                                                     pass_views.iloc[2], column_name, is_ground_truth):
            if box not in pass_boxes:
                pass_boxes.append(box)
        for box, label, score in self._check_three_consecutive_views(pass_views.iloc[1], pass_views.iloc[0],
                                                                     pass_views.iloc[2], column_name, is_ground_truth):
            if box not in pass_boxes:
                pass_boxes.append(box)
        for box, label, score in self._check_three_consecutive_views(pass_views.iloc[1], pass_views.iloc[2],
                                                                     pass_views.iloc[3], column_name, is_ground_truth):
            if box not in pass_boxes:
                pass_boxes.append(box)
        for box, label, score in self._check_three_consecutive_views(pass_views.iloc[2], pass_views.iloc[0],
                                                                     pass_views.iloc[1], column_name, is_ground_truth):
            if box not in pass_boxes:
                pass_boxes.append(box)
        for box, label, score in self._check_three_consecutive_views(pass_views.iloc[2], pass_views.iloc[1],
                                                                     pass_views.iloc[3], column_name, is_ground_truth):
            if box not in pass_boxes:
                pass_boxes.append(box)
        for box, label, score in self._check_three_consecutive_views(pass_views.iloc[2], pass_views.iloc[3],
                                                                     pass_views.iloc[4], column_name, is_ground_truth):
            if box not in pass_boxes:
                pass_boxes.append(box)
        for box, label, score in self._check_three_consecutive_views(pass_views.iloc[3], pass_views.iloc[1],
                                                                     pass_views.iloc[2], column_name, is_ground_truth):
            if box not in pass_boxes:
                pass_boxes.append(box)
        for box, label, score in self._check_three_consecutive_views(pass_views.iloc[3], pass_views.iloc[2],
                                                                     pass_views.iloc[4], column_name, is_ground_truth):
            if box not in pass_boxes:
                pass_boxes.append(box)
        for box, label, score in self._check_three_consecutive_views(pass_views.iloc[3], pass_views.iloc[4],
                                                                     pass_views.iloc[5], column_name, is_ground_truth):
            if box not in pass_boxes:
                pass_boxes.append(box)
        for box, label, score in self._check_three_consecutive_views(pass_views.iloc[4], pass_views.iloc[3],
                                                                     pass_views.iloc[2], column_name, is_ground_truth):
            if box not in pass_boxes:
                pass_boxes.append(box)
        for box, label, score in self._check_three_consecutive_views(pass_views.iloc[4], pass_views.iloc[3],
                                                                     pass_views.iloc[5], column_name, is_ground_truth):
            if box not in pass_boxes:
                pass_boxes.append(box)
        for box, label, score in self._check_three_consecutive_views(pass_views.iloc[5], pass_views.iloc[3],
                                                                     pass_views.iloc[4], column_name, is_ground_truth):
            if box not in pass_boxes:
                pass_boxes.append(box)

        if not is_ground_truth:
            return pass_boxes, pass_labels, pass_scores
        else:
            return pass_boxes, pass_labels

    @staticmethod
    def _extract_file_name_with_manipulation_prefix(full_file_name: str):
        file_name = full_file_name[:14]
        if "vchw" in full_file_name:
            manipulation_prefix = "vchw"
        elif "vc" in full_file_name:
            manipulation_prefix = "vc"
        elif "hw" in full_file_name:
            manipulation_prefix = "hw"
        else:
            manipulation_prefix = ""

        return file_name, manipulation_prefix

    @staticmethod
    def _sort_df_by_map(df: pd.DataFrame):
        index_map = [0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 3, 4, 5,
                     6, 7, 8, 9]

        return df.set_index(pd.Index(index_map)).sort_index()

    @staticmethod
    def _handle_existing_file(log_dir: str, file_name: str):
        path = os.path.join(log_dir, "pallet_level_precision_recall")
        os.makedirs(path, exist_ok=True)

        full_path = os.path.join(path, f"{file_name}.csv")

        if os.path.exists(full_path):
            suffix = int(time.time())
            shutil.copy2(full_path, os.path.join(path, f"{file_name}_{suffix}.csv"))

            os.remove(full_path)

        return path, full_path

    @staticmethod
    def _calc_iou(gt_bbox, pred_bbox):
        """
        This function takes the predicted bounding box and ground truth bounding box and
        return the IoU ratio
        """
        x_top_left_gt, y_top_left_gt, x_bottom_right_gt, y_bottom_right_gt = gt_bbox
        x_top_left_p, y_top_left_p, x_bottom_right_p, y_bottom_right_p = pred_bbox

        if (x_top_left_gt > x_bottom_right_gt) or (y_top_left_gt > y_bottom_right_gt):
            raise AssertionError("Ground Truth Bounding Box is not correct")
        if (x_top_left_p > x_bottom_right_p) or (y_top_left_p > y_bottom_right_p):
            raise AssertionError("Predicted Bounding Box is not correct", x_top_left_p, x_bottom_right_p, y_top_left_p,
                                 y_bottom_right_gt)

        # if the GT bbox and predicted BBox do not overlap then iou=0
        # If bottom right of x-coordinate GT bbox is less than or above the top left of x coordinate of the predicted
        # BBox
        if x_bottom_right_gt < x_top_left_p:
            return 0.0
        # If bottom right of y-coordinate GT bbox is less than or above the top left of y coordinate of the predicted
        # BBox
        if y_bottom_right_gt < y_top_left_p:
            return 0.0
        # If bottom right of x-coordinate GT bbox is greater than or below the bottom right of x coordinate of the
        # predicted BBox
        if x_top_left_gt > x_bottom_right_p:
            return 0.0
        # If bottom right of y-coordinate GT bbox is greater than or below the bottom right of y coordinate of the
        # predicted BBox
        if y_top_left_gt > y_bottom_right_p:
            return 0.0

        gt_bbox_area = (x_bottom_right_gt - x_top_left_gt + 1) * (y_bottom_right_gt - y_top_left_gt + 1)
        pred_bbox_area = (x_bottom_right_p - x_top_left_p + 1) * (y_bottom_right_p - y_top_left_p + 1)

        x_top_left = np.max([x_top_left_gt, x_top_left_p])
        y_top_left = np.max([y_top_left_gt, y_top_left_p])
        x_bottom_right = np.min([x_bottom_right_gt, x_bottom_right_p])
        y_bottom_right = np.min([y_bottom_right_gt, y_bottom_right_p])

        intersection_area = (x_bottom_right - x_top_left + 1) * (y_bottom_right - y_top_left + 1)

        union_area = (gt_bbox_area + pred_bbox_area - intersection_area)

        return intersection_area / union_area

    def _get_vote(self, coordinates, next_view, column_name, iou_threshold: float = 0.7):
        local_vote = 0

        for x in range(len(next_view[column_name])):
            next_box = next_view[column_name][x]
            next_coordinates = [
                int(next_box[0]),
                int(next_box[1]),
                int(next_box[2]),
                int(next_box[3])
            ]
            voting_iou = self._calc_iou(coordinates, next_coordinates)

            if voting_iou >= iou_threshold:
                local_vote += 1

        return local_vote

    def _check_three_consecutive_views(self, current_view, first_view, second_view, column_name, is_ground_truth: bool):
        picked_boxes = []
        picked_labels = []
        picked_scores = []
        # TODO: add labels and scores to the voting
        for t in range(len(current_view[column_name])):
            vote = 0
            current_box = current_view[column_name][t]
            current_coordinates = [
                int(current_box[0]),
                int(current_box[1]),
                int(current_box[2]),
                int(current_box[3])
            ]

            for view in (first_view, second_view):
                if self._get_vote(current_coordinates, view, column_name=column_name) > 0:
                    vote += 1

            if vote == 2:
                picked_boxes.append(current_box)

        return picked_boxes, picked_labels, picked_scores

    @staticmethod
    def _merge_bounding_boxes(bounding_boxes):
        if len(bounding_boxes) == 0:
            return [], []
        bboxes = np.array(bounding_boxes)

        min_x1, min_y1 = np.minimum([bboxes[0][0], bboxes[0][1]], [bboxes[1][0], bboxes[1][1]])
        max_x2, max_y2 = np.maximum([bboxes[0][2], bboxes[0][3]], [bboxes[1][2], bboxes[1][3]])

        average_score = (bboxes[0][4] + bboxes[1][4]) / 2

        return [int(min_x1), int(min_y1), int(max_x2), int(max_y2), float(average_score)]
