from os import listdir
from os.path import join

from torch.utils.data import ConcatDataset

from data.astro_pallet import AstroPallet


class AstroDataset:
    astro_dataset = []

    def __init__(self, root, annotation_file_name, transform=None, max_pallets_to_load: int = 0):
        self.root = root
        self.annotation_file_name = annotation_file_name
        self.transform = transform
        self.max_pallets_to_load = max_pallets_to_load

    def get_concatenated_datasets(self):
        dataset = []
        for view in self._get_directory_tree_list():
            dataset.append(AstroPallet(view, self.annotation_file_name, self.transform))

        self.astro_dataset = ConcatDataset(dataset)

        return self.astro_dataset

    def _get_directory_tree_list(self):
        sub_folders = []
        folder_count = 0
        for folder in listdir(self.root):
            if self.max_pallets_to_load > 0:
                if folder_count == self.max_pallets_to_load:
                    break
            for sub_folder in listdir(join(self.root, folder)):
                if self.annotation_file_name in listdir(join(self.root, folder, sub_folder)):
                    sub_folders.append(join(self.root, folder, sub_folder))

            folder_count += 1

        return sub_folders
