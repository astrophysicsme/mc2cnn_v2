import torch

from typing import Union

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from data.__init__ import MC2CNNDataModule
from model import MC2CNN
from callbacks import callbacks

if __name__ == '__main__':
    # TODO: place the below configurations in config files. probably use hydra
    max_pallets_to_load: int = 1  # use int <= 0 to load all available pallets
    run_training: bool = True
    run_testing: bool = True

    mc2cnn_data_directory: str = "datasets"
    mc2cnn_data_annotation_file_name: str = "_annotation.coco.json"
    mc2cnn_data_num_workers: int = 0
    mc2cnn_data_batch_size: int = 2

    mc2cnn_max_image_size: int = 1950

    mc2cnn_num_classes: int = 12
    mc2cnn_learning_rate: float = 0.0001
    mc2cnn_weight_decay: float = 0.005
    mc2cnn_momentum: float = 0.9

    mc2cnn_box_nms_threshold: float = 0.3

    trainer_num_gpus: int = min(1, torch.cuda.device_count())
    trainer_epochs: int = 100
    trainer_log_every_n_step: int = 1
    trainer_precision: int = 16

    tensorboard_logger_save_dir: str = "./"
    tensorboard_logger_version: Union[int, None] = None

    checkpoint_path: Union[str, None] = None
    # checkpoint_path: Union[str, None] = "lightning_logs/version_0/checkpoints/epoch=28-step=171448.ckpt"

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

    if tensorboard_logger_version is None:
        logger = TensorBoardLogger(save_dir=tensorboard_logger_save_dir)
    else:
        logger = TensorBoardLogger(save_dir=tensorboard_logger_save_dir, version=tensorboard_logger_version)

    # Initialize a trainer
    trainer = Trainer(
        max_epochs=trainer_epochs,
        gpus=trainer_num_gpus,
        log_every_n_steps=trainer_log_every_n_step,
        precision=trainer_precision,
        logger=logger,
        callbacks=callbacks,
        weights_summary="full"
    )

    trainer.tune(model=mc2cnn, datamodule=mc2cnn_data_module)

    # Train the model
    if run_training:
        if checkpoint_path is None:
            trainer.fit(model=mc2cnn, datamodule=mc2cnn_data_module)
        else:
            trainer.fit(model=mc2cnn, datamodule=mc2cnn_data_module,
                        ckpt_path=checkpoint_path)

    # TODO: Hyperparameter tuning

    if run_testing:
        if checkpoint_path is None:
            trainer.test(model=mc2cnn, datamodule=mc2cnn_data_module)
        else:
            trainer.test(model=mc2cnn, datamodule=mc2cnn_data_module, ckpt_path=checkpoint_path)
