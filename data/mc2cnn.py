from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule

from data.astro_dataset import AstroDataset
from data.utils import collate_fn

from transformations import transform


class MC2CNNDataModule(LightningDataModule):
    def __init__(self, data_dir, annotation_file_name="_annotation.coco.json", num_workers=0, batch_size=4,
                 shuffle_train_dataloader=True, shuffle_val_dataloader=False, shuffle_test_dataloader=False,
                 max_pallets_to_load=0):
        super().__init__()
        # Set our init args as class attributes
        self.max_pallets_to_load = max_pallets_to_load

        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size

        self.annotation_file_name = annotation_file_name

        self.train_transform = transform(stage="train")
        self.val_transform = transform(stage="val")
        self.test_transform = transform(stage="test")

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.shuffle_train_dataloader = shuffle_train_dataloader
        self.shuffle_val_dataloader = shuffle_val_dataloader
        self.shuffle_test_dataloader = shuffle_test_dataloader

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage in (None, "fit"):
            self.train_dataset = AstroDataset(self.data_dir + "/train", self.annotation_file_name,
                                              transform=self.train_transform,
                                              max_pallets_to_load=self.max_pallets_to_load).get_concatenated_datasets()
            self.val_dataset = AstroDataset(self.data_dir + "/val", self.annotation_file_name,
                                            transform=self.val_transform,
                                            max_pallets_to_load=self.max_pallets_to_load).get_concatenated_datasets()

        # Assign test dataset for use in dataloader(s)
        if stage in (None, "test"):
            self.test_dataset = AstroDataset(self.data_dir + "/test", self.annotation_file_name,
                                             transform=self.test_transform,
                                             max_pallets_to_load=self.max_pallets_to_load).get_concatenated_datasets()

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=self.shuffle_train_dataloader, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=self.shuffle_val_dataloader, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          shuffle=self.shuffle_test_dataloader, collate_fn=collate_fn)
