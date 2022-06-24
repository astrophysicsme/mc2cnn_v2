import os
from typing import Optional


class PalletLevelPrecisionRecall:
    def __init__(
            self,
            file_name: str,
            log_dir: Optional[str] = "astro_metrics_logs",
    ):
        self._handle_existing_file(log_dir=log_dir, file_name=file_name)
        self.log_dir = log_dir



        self.file_name = file_name







    def update(
            self,
            image_file_names,
            targets,
            labels,
            minimized_pred_boxes
    ):
        # TODO: log truth table in csv format in the experiment folder
        pass

    def compute(self) -> dict:
        precision = None
        recall = None
        return {
            "precision": precision,
            "recall": recall
        }

    def _handle_existing_file(self, log_dir: str, file_name: str):
        path = os.path.join(log_dir, "pallet_level_precision_recall")
        os.makedirs(path, exist_ok=False)

        self.path = path
        self.full_path = os.path.join(self.path, f"{file_name}.csv")

        if os.path.exists(self.full_path):
            # backup the file
            pass


