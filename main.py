import torch

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, TQDMProgressBar

from data.mc2cnn import MC2CNNDataModule
from model import MC2CNN

if __name__ == '__main__':
    # TODO: place the below configurations in config files. probably use hydra
    max_pallets_to_load = 0  # use int <= 0 to load all available pallets

    mc2cnn_data_directory = "datasets"
    mc2cnn_data_annotation_file_name = "_annotation.coco.json"
    mc2cnn_data_num_workers = 0
    mc2cnn_data_batch_size = 4

    mc2cnn_num_classes = 12
    mc2cnn_learning_rate = 5e-4  # this learning rate will be overwritten by the auto learning rate finder

    trainer_enable_auto_lr_finder = True
    trainer_num_gpus = min(1, torch.cuda.device_count())
    trainer_epochs = 100
    trainer_log_every_n_step = 1
    trainer_precision = 16

    learning_rate_logging_interval = "epoch"

    tqdm_progress_bar_refresh_rate = 1

    early_stopping_monitor = "val_accuracy"
    early_stopping_min_delta = 0.00
    early_stopping_patience = 3
    early_stopping_verbose = True
    early_stopping_mode = "max"

    torch.cuda.empty_cache()

    # Init our data module
    mc2cnn_data_module = MC2CNNDataModule(data_dir=mc2cnn_data_directory,
                                          annotation_file_name=mc2cnn_data_annotation_file_name,
                                          batch_size=mc2cnn_data_batch_size, num_workers=mc2cnn_data_num_workers,
                                          max_pallets_to_load=max_pallets_to_load)

    # Init our model
    # mc2cnn18 = MC2CNN(resnet_name="resnet18", n_classes=mc2cnn_num_classes, lr_rate=mc2cnn_learning_rate,
    #                   batch_size=mc2cnn_data_batch_size)
    # mc2cnn34 = MC2CNN(resnet_name="resnet34", n_classes=mc2cnn_num_classes, lr_rate=mc2cnn_learning_rate,
    #                   batch_size=mc2cnn_data_batch_size)
    mc2cnn50 = MC2CNN(resnet_name="resnet50", n_classes=mc2cnn_num_classes, lr_rate=mc2cnn_learning_rate,
                      batch_size=mc2cnn_data_batch_size)
    # mc2cnn101 = MC2CNN(resnet_name="resnet101", n_classes=mc2cnn_num_classes, lr_rate=mc2cnn_learning_rate,
    #                    batch_size=mc2cnn_data_batch_size)
    # mc2cnn152 = MC2CNN(resnet_name="resnet152", n_classes=mc2cnn_num_classes, lr_rate=mc2cnn_learning_rate,
    #                    batch_size=mc2cnn_data_batch_size)

    # Initialize a trainer
    trainer = Trainer(
        # auto_lr_find=trainer_enable_auto_lr_finder,
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

    # finds learning rate automatically
    # trainer.tune(model=mc2cnn50, datamodule=mc2cnn_data_module)

    # TODO: log more variables during training and validating
    # TODO: check if we can draw plots during training and validating
    # Train the model
    trainer.fit(model=mc2cnn50, datamodule=mc2cnn_data_module)

    # TODO: Hyperparameter tuning

    # TODO: log more variables during testing
    # TODO: check if we can draw plots during testing
    # trainer.test(model=mc2cnn152, ckpt_path="best", datamodule=mc2cnn_data_module)
