import os
import shutil
import time
from ast import literal_eval

import pandas as pd
import numpy as np
from typing import Optional


class PalletLevelPrecisionRecall:
    def __init__(
            self,
            log_dir: Optional[str] = "astro_metrics_logs",
            pallet_manipulations_identifiers: Optional[tuple] = ("_vchw.", "_hw.", "_vc.", ""),
            pallet_manipulations: Optional[int] = 4,
            views_per_pallet: Optional[int] = 30,
            passes_per_pallet: Optional[int] = 5,
            views_per_pass: Optional[int] = 6,
            detection_threshold: Optional[float] = 0.75,
    ):
        self._log_dir = log_dir
        self._image_level_file_name = "image_level_truth_table"
        self._single_level_tt_path, self._single_level_tt_full_path = self._backup_old_file(self._log_dir,
                                                                                            self._image_level_file_name)
        self._image_level_tt_columns = [
            "image_file_name",
            "gt_boxes",
            "gt_labels",
            "pred_boxes",
            "pred_labels",
            "pred_scores"
        ]

        self._pass_level_file_name = "pass_level_truth_table"
        self._pass_level_tt_path, self._pass_level_tt_full_path = self._backup_old_file(self._log_dir,
                                                                                        self._pass_level_file_name)
        self._pass_level_tt_columns = [
            "image_file_name",
            "manipulation",
            "views",
            "gt_boxes",
            "gt_labels",
            "gt_views",
            "pred_boxes",
            "pred_labels",
            "pred_scores",
            "pred_views",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
        ]

        self._summary_file_name = "summary"
        self._summary_path, self._summary_full_path = self._backup_old_file(self._log_dir, self._summary_file_name)
        self._summary_columns = [
            "metric",
            "value"
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

        df = pd.DataFrame(data=data, columns=self._image_level_tt_columns)
        if os.path.exists(self._single_level_tt_full_path):
            original_df = pd.read_csv(self._single_level_tt_full_path)
            df = pd.concat([original_df, df], ignore_index=True)

        df.to_csv(self._single_level_tt_full_path, index=False)

    def compute(self):
        if not os.path.exists(self._single_level_tt_full_path):
            raise FileNotFoundError
        else:
            assert self._pallet_manipulations == len(self._pallet_manipulations_identifiers)

            pass_level_truth_table = []
            for chunk in pd.read_csv(self._single_level_tt_full_path, converters={
                "gt_boxes": literal_eval,
                "gt_labels": literal_eval,
                "pred_boxes": literal_eval,
                "pred_labels": literal_eval,
                "pred_scores": literal_eval,
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
                    #   keep the pred boxes, label and scores, that overlap in 3 consecutive views
                    for i in range(start_position, self._views_per_pallet, self._views_per_pass):
                        next_position = i + self._views_per_pass

                        views = pallet[i:next_position]

                        view_range = f"{i} to {next_position - 1}"
                        file_name, manipulation_prefix = self._extract_file_name_with_manipulation_prefix(
                            views.iloc[0]["image_file_name"])

                        gt_boxes, gt_labels, gt_views = self._single_view_to_p_view_t_t(views, "gt", True)
                        p_boxes, p_labels, p_views, p_scores = self._single_view_to_p_view_t_t(views, "pred", False)

                        pass_level_truth_table.append({
                            "image_file_name": file_name,
                            "manipulation": manipulation_prefix,
                            "views": view_range,
                            "gt_boxes": gt_boxes,
                            "gt_labels": gt_labels,
                            "gt_views": gt_views,
                            "pred_boxes": p_boxes,
                            "pred_labels": p_labels,
                            "pred_scores": p_scores,
                            "pred_views": p_views,
                        })

            precisions = []
            recalls = []
            pass_results = []
            for pass_result in pass_level_truth_table:
                p_boxes = pass_result["pred_boxes"]
                p_labels = pass_result["pred_labels"]
                p_views = pass_result["pred_views"]
                gt_boxes = pass_result["gt_boxes"]
                gt_labels = pass_result["gt_labels"]
                gt_views = pass_result["gt_views"]
                pass_result["tp"] = 0
                pass_result["fp"] = 0
                pass_result["fn"] = 0

                number_of_threats = 0
                for x in gt_boxes:
                    number_of_threats += len(x)

                for pbs in range(len(p_boxes)):
                    for pb in range(len(p_boxes[pbs])):
                        hit = 0
                        for gbs in range(len(gt_boxes)):
                            for gb in range(len(gt_boxes[gbs])):
                                if p_views[pbs][pb] == gt_views[gbs][gb]:
                                    if p_labels[pbs][pb] == gt_labels[gbs][gb]:
                                        if self._calc_iou(gt_boxes[gbs][gb], p_boxes[pbs][pb]) >= 0.3:
                                            hit += 1
                        if hit > 0:
                            pass_result["tp"] += 1
                        else:
                            pass_result["fp"] += 1

                fn = number_of_threats - pass_result["tp"]
                pass_result["fn"] = fn if fn >= 0 else 0
                tp_fp = pass_result["tp"] + pass_result["fp"]
                pass_result["precision"] = (pass_result["tp"] / tp_fp) if tp_fp != 0 else 0
                tp_fn = pass_result["tp"] + pass_result["fn"]
                pass_result["recall"] = (pass_result["tp"] / tp_fn) if tp_fn != 0 else 0
                precisions.append(pass_result["precision"])
                recalls.append(pass_result["recall"])
                pass_results.append(pass_result)

            pass_level_tt_result_df = pd.DataFrame(data=pass_level_truth_table, columns=self._pass_level_tt_columns)
            pass_level_tt_result_df.to_csv(self._pass_level_tt_full_path, index=False)

            summary = {
                "mean avg. precision": sum(precisions) / len(precisions),
                "mean avg. recall": sum(recalls) / len(recalls),
                "mean avg. false alarm rate": 1 - (sum(recalls) / len(recalls))
            }

            summary_df_data = []
            for k, v in summary.items():
                summary_df_data.append({"metric": k, "value": v})

            summary_df = pd.DataFrame(data=summary_df_data, columns=self._summary_columns)
            summary_df.to_csv(self._summary_full_path, index=False)

            return summary

    def _single_view_to_p_view_t_t(self, views, col_pref, is_gt=False):
        pass_boxes = []
        pass_labels = []
        pass_scores = []
        pass_views = []

        check_1 = self._check_three_consecutive_views(views.iloc[0], views.iloc[1], views.iloc[2], col_pref, is_gt)

        check_2 = self._check_three_consecutive_views(views.iloc[1], views.iloc[0], views.iloc[2], col_pref, is_gt)
        check_3 = self._check_three_consecutive_views(views.iloc[1], views.iloc[2], views.iloc[3], col_pref, is_gt)

        check_4 = self._check_three_consecutive_views(views.iloc[2], views.iloc[0], views.iloc[1], col_pref, is_gt)
        check_5 = self._check_three_consecutive_views(views.iloc[2], views.iloc[1], views.iloc[3], col_pref, is_gt)
        check_6 = self._check_three_consecutive_views(views.iloc[2], views.iloc[3], views.iloc[4], col_pref, is_gt)

        check_7 = self._check_three_consecutive_views(views.iloc[3], views.iloc[1], views.iloc[2], col_pref, is_gt)
        check_8 = self._check_three_consecutive_views(views.iloc[3], views.iloc[2], views.iloc[4], col_pref, is_gt)
        check_9 = self._check_three_consecutive_views(views.iloc[3], views.iloc[4], views.iloc[5], col_pref, is_gt)

        check_10 = self._check_three_consecutive_views(views.iloc[4], views.iloc[3], views.iloc[2], col_pref, is_gt)
        check_11 = self._check_three_consecutive_views(views.iloc[4], views.iloc[3], views.iloc[5], col_pref, is_gt)

        check_12 = self._check_three_consecutive_views(views.iloc[5], views.iloc[3], views.iloc[4], col_pref, is_gt)

        res = [
            check_1,
            self._merge_single_views_bounding_boxes([check_2, check_3], is_gt),
            self._merge_single_views_bounding_boxes([check_4, check_5, check_6], is_gt),
            self._merge_single_views_bounding_boxes([check_7, check_8, check_9], is_gt),
            self._merge_single_views_bounding_boxes([check_10, check_11], is_gt),
            check_12
        ]

        for r in range(len(res)):
            if res[r][0] not in pass_boxes and res[r][0] != []:
                pass_boxes.append(res[r][0])
                pass_labels.append(res[r][1])
                pass_views.append(res[r][2])
                if not is_gt:
                    pass_scores.append(res[r][3])

        if not is_gt:
            return pass_boxes, pass_labels, pass_views, pass_scores
        else:
            return pass_boxes, pass_labels, pass_views

    def _merge_single_views_bounding_boxes(self, combined_checks, is_gt):
        combined_results = []
        for res in range(len(combined_checks)):
            if combined_results == list():
                combined_results.append(combined_checks[res][0])
                combined_results.append(combined_checks[res][1])
                combined_results.append(combined_checks[res][2])
                combined_results.append(combined_checks[res][3])
            else:
                for c_c in range(len(combined_checks[res][0])):
                    box_found = 0
                    for c_r in range(len(combined_results[0])):
                        if combined_checks[res][1][c_c] == combined_results[1][c_r]:
                            if self._calc_iou(combined_checks[res][0][c_c], combined_results[0][c_r]) >= 0.6:
                                combined_results[0][c_r] = self._merge_bounding_boxes(combined_checks[res][0][c_c],
                                                                                      combined_results[0][c_r])
                                box_found += 1
                    if box_found == 0:
                        combined_results[0].append(combined_checks[res][0][c_c])
                        combined_results[1].append(combined_checks[res][1][c_c])
                        combined_results[2].append(combined_checks[res][2][c_c])
                        if not is_gt:
                            combined_results[3].append(combined_checks[res][3][c_c])

        return combined_results

    @staticmethod
    def _extract_file_name_with_manipulation_prefix(full_file_name: str):
        file_name = full_file_name[:15]
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
    def _backup_old_file(log_dir: str, file_name: str):
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
        This function takes the pred bounding box and ground truth bounding box and
        return the IoU ratio
        """
        x_t_l_gt, y_t_l_gt, x_b_r_gt, y_b_r_gt = gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3]
        x_t_l_p, y_t_l_p, x_b_r_p, y_b_r_p = pred_bbox[0], pred_bbox[1], pred_bbox[2], pred_bbox[3]

        if (x_t_l_gt > x_b_r_gt) or (y_t_l_gt > y_b_r_gt):
            raise AssertionError("Ground Truth Bounding Box is not correct")
        if (x_t_l_p > x_b_r_p) or (y_t_l_p > y_b_r_p):
            raise AssertionError("pred Bounding Box is not correct", x_t_l_p, x_b_r_p, y_t_l_p, y_b_r_gt)

        # if the GT bbox and pred BBox do not overlap then iou=0
        # If bottom right of x GT bbox is less than or above the top left of x of the pred Box
        if x_b_r_gt < x_t_l_p:
            return 0.0
        # If bottom right of y GT bbox is less than or above the top left of y of the pred BBox
        if y_b_r_gt < y_t_l_p:
            return 0.0
        # If bottom right of x GT bbox is greater than or below the bottom right of x of the pred BBox
        if x_t_l_gt > x_b_r_p:
            return 0.0
        # If bottom right of y GT bbox is greater than or below the bottom right of y of the pred BBox
        if y_t_l_gt > y_b_r_p:
            return 0.0

        gt_bbox_area = (x_b_r_gt - x_t_l_gt + 1) * (y_b_r_gt - y_t_l_gt + 1)
        pred_bbox_area = (x_b_r_p - x_t_l_p + 1) * (y_b_r_p - y_t_l_p + 1)

        x_t_l = np.max([x_t_l_gt, x_t_l_p])
        y_t_l = np.max([y_t_l_gt, y_t_l_p])
        x_b_r = np.min([x_b_r_gt, x_b_r_p])
        y_b_r = np.min([y_b_r_gt, y_b_r_p])

        intersection_area = (x_b_r - x_t_l + 1) * (y_b_r - y_t_l + 1)

        union_area = (gt_bbox_area + pred_bbox_area - intersection_area)

        return intersection_area / union_area

    def _get_vote(self, coordinates, next_view, col_pref, iou_threshold: float = 0.7):
        local_vote = 0

        for x in range(len(next_view[f"{col_pref}_boxes"])):
            next_box = next_view[f"{col_pref}_boxes"][x]
            next_coordinates = [
                int(next_box[0]),
                int(next_box[1]),
                int(next_box[2]),
                int(next_box[3])
            ]
            if int(next_view[f"{col_pref}_labels"][x]) != coordinates[4]:
                return local_vote

            voting_iou = self._calc_iou(coordinates, next_coordinates)

            if voting_iou >= iou_threshold:
                local_vote += 1

        return local_vote

    def _check_three_consecutive_views(self, current_view, first_view, second_view, col_pref, is_gt: bool):
        picked_boxes = []
        picked_labels = []
        picked_scores = []
        picked_views = []

        view_number = int(str(current_view["image_file_name"][16:18]).rstrip('.').rstrip('_'))

        for t in range(len(current_view[f"{col_pref}_boxes"])):
            vote = 0
            current_box = current_view[f"{col_pref}_boxes"][t]
            current_coordinates = [
                int(current_box[0]),
                int(current_box[1]),
                int(current_box[2]),
                int(current_box[3]),
                int(current_view[f"{col_pref}_labels"][t]),
            ]

            if not is_gt:
                current_coordinates.append(float(current_view[f"{col_pref}_scores"][t]))

            for view in (first_view, second_view):
                if self._get_vote(current_coordinates, view, col_pref=col_pref, iou_threshold=0.3) > 0:
                    vote += 1

            if vote == 2:
                picked_boxes.append(current_box)
                picked_labels.append(current_view[f"{col_pref}_labels"][t])
                picked_views.append(view_number)
                if not is_gt:
                    picked_scores.append(current_view[f"{col_pref}_scores"][t])

        return picked_boxes, picked_labels, picked_views, picked_scores

    @staticmethod
    def _merge_bounding_boxes(gt_box, pred_box):
        if len(gt_box) == 0 or len(pred_box) == 0:
            return []
        min_x1, min_y1 = np.minimum([gt_box[0], gt_box[1]], [pred_box[0], pred_box[1]])
        max_x2, max_y2 = np.maximum([gt_box[2], gt_box[3]], [pred_box[2], pred_box[3]])

        return [int(min_x1), int(min_y1), int(max_x2), int(max_y2)]
