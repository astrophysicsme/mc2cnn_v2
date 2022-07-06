from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, TQDMProgressBar, ModelCheckpoint


# TODO: place the below configurations in config files. probably use hydra
learning_rate_logging_interval: str = "epoch"

tqdm_progress_bar_refresh_rate: int = 24

early_stopping_monitor: str = "val_accuracy"
early_stopping_min_delta: float = 0.00
early_stopping_patience: int = 3
early_stopping_verbose: bool = False
early_stopping_mode: str = "max"

model_checkpoint_save_top_k: int = 5
model_checkpoint_monitor: str = "val_accuracy"
model_checkpoint_mode: str = "max"

callbacks = [
    LearningRateMonitor(logging_interval=learning_rate_logging_interval),
    TQDMProgressBar(refresh_rate=tqdm_progress_bar_refresh_rate),
    EarlyStopping(monitor=early_stopping_monitor, min_delta=early_stopping_min_delta,
                  patience=early_stopping_patience, verbose=early_stopping_verbose, mode=early_stopping_mode),
    ModelCheckpoint(save_top_k=model_checkpoint_save_top_k, monitor=model_checkpoint_monitor,
                    mode=model_checkpoint_mode)
]
