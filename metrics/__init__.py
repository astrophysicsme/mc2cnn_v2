import os
import shutil
import time

import pandas as pd
from typing import Optional


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
    ):
        self.log_dir = log_dir
        self.file_name = file_name
        self.pr_path, self.pr_full_path = self._handle_existing_file(log_dir=self.log_dir, file_name=self.file_name)
        self.pr_columns = [
            "image_file_name",
            "gt_boxes",
            "gt_labels",
            "predicted_boxes",
            "predicted_labels",
            "predicted_scores"
        ]

        self.tt_path, self.tt_full_path = self._handle_existing_file(log_dir=self.log_dir, file_name="truth_table")
        self.tt_columns = [
            "metric name",
            "metric value"
        ]

        self.pallet_manipulations = pallet_manipulations
        self.views_per_pallet = views_per_pallet
        self.passes_per_pallet = passes_per_pallet
        self.views_per_pass = views_per_pass
        self.pallet_manipulations_identifiers = pallet_manipulations_identifiers

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

        df = pd.DataFrame(data=data, columns=self.pr_columns)
        if os.path.exists(self.pr_full_path):
            original_df = pd.read_csv(self.pr_full_path)
            df = pd.concat([original_df, df], ignore_index=True)

        df.to_csv(self.pr_full_path, index=False)

    def compute(self):
        if os.path.exists(self.pr_full_path):
            chunk_size = self.pallet_manipulations * self.views_per_pallet

            assert self.pallet_manipulations == len(self.pallet_manipulations_identifiers)

            for chunk in pd.read_csv(self.pr_full_path, chunksize=chunk_size):
                # separate all the different manipulations
                pallets_sorted_by_manipulation = list()
                for identifier in self.pallet_manipulations_identifiers:
                    if identifier != "":
                        sub = chunk["image_file_name"].str.contains(identifier)
                        pallets_sorted_by_manipulation.append(chunk[sub])
                        # drop all extracted rows from the main pd chunk
                        chunk = chunk[sub == False]
                    else:
                        pallets_sorted_by_manipulation.append(chunk)
                # order the rows according to the pallets names. pallets are ordered alphabetically not by view
                # loop over the views from the same pass
                #   keep the predicted boxes, label and scores, that overlap in 3 consecutive views
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
                self.tt_columns[0]: list(final_result.keys()),
                self.tt_columns[1]: list(final_result.values())
            }

            tt_result_df = pd.DataFrame(data=final_result_for_pd_df, columns=self.tt_columns)
            tt_result_df.to_csv(self.tt_full_path, index=False)

            return final_result

        else:
            raise FileNotFoundError

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
