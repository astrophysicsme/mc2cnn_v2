import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, TQDMProgressBar

from data.__init__ import MC2CNNDataModule
from model import MC2CNN

if __name__ == '__main__':
    # TODO: place the below configurations in config files. probably use hydra
    max_pallets_to_load = 0  # use int <= 0 to load all available pallets

    mc2cnn_data_directory = "datasets"
    mc2cnn_data_annotation_file_name = "_annotation.coco.json"
    mc2cnn_data_num_workers = 0
    mc2cnn_data_batch_size = 2

    mc2cnn_max_image_size = 1950

    mc2cnn_num_classes = 12
    mc2cnn_learning_rate = 0.0001
    mc2cnn_weight_decay = 0.005
    mc2cnn_momentum = 0.9

    mc2cnn_box_nms_threshold = 0.3

    trainer_num_gpus = min(1, torch.cuda.device_count())
    trainer_epochs = 100
    trainer_log_every_n_step = 1
    trainer_precision = 16

    learning_rate_logging_interval = "epoch"

    tqdm_progress_bar_refresh_rate = 1

    early_stopping_monitor = "val_accuracy"
    early_stopping_min_delta = 0.00
    early_stopping_patience = 5
    early_stopping_verbose = False
    early_stopping_mode = "max"

    torch.cuda.empty_cache()

    # Init our data module
    mc2cnn_data_module = MC2CNNDataModule(data_dir=mc2cnn_data_directory,
                                          annotation_file_name=mc2cnn_data_annotation_file_name,
                                          batch_size=mc2cnn_data_batch_size, num_workers=mc2cnn_data_num_workers,
                                          max_pallets_to_load=max_pallets_to_load)

    # Init our model
    mc2cnn = MC2CNN(resnet_name="resnet152", n_classes=mc2cnn_num_classes, lr_rate=mc2cnn_learning_rate,
                    batch_size=mc2cnn_data_batch_size, box_nms_threshold=mc2cnn_box_nms_threshold,
                    weight_decay=mc2cnn_weight_decay, momentum=mc2cnn_momentum, max_image_size=mc2cnn_max_image_size)

    # Initialize a trainer
    trainer = Trainer(
        max_epochs=trainer_epochs,
        gpus=trainer_num_gpus,
        log_every_n_steps=trainer_log_every_n_step,
        precision=trainer_precision,
        callbacks=[
            LearningRateMonitor(logging_interval=learning_rate_logging_interval),
            TQDMProgressBar(refresh_rate=tqdm_progress_bar_refresh_rate),
            EarlyStopping(monitor=early_stopping_monitor, min_delta=early_stopping_min_delta,
                          patience=early_stopping_patience, verbose=early_stopping_verbose, mode=early_stopping_mode),
        ]
    )

    trainer.tune(model=mc2cnn, datamodule=mc2cnn_data_module)

    # Train the model
    trainer.fit(model=mc2cnn, datamodule=mc2cnn_data_module)

    # TODO: Hyperparameter tuning

    # trainer.test(model=mc2cnn152, ckpt_path="best", datamodule=mc2cnn_data_module)
