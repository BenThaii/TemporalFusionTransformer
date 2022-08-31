import copy
from pathlib import Path
import warnings
import argparse
import json
import os
import logging
import sys

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.distributed as dist

from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))



def _train(args):
    global logger
    
    # unpack arguments:
    max_prediction_length = args.max_prediction_length
    max_encoder_length = args.max_encoder_length
    num_epochs = args.num_epochs
    early_stopping_patience = args.early_stopping_patience
    multiprocessing_workers = args.multiprocessing_workers


    dropout_rate = args.dropout_rate
    hidden_layer_size = args.hidden_layer_size
    learning_rate = args.learning_rate
    minibatch_size = args.minibatch_size
    max_gradient_norm = args.max_gradient_norm
    num_heads = args.num_heads
    
    is_distributed = len(args.hosts) > 1 and args.dist_backend is not None
    logger.debug("Distributed training - {}".format(is_distributed))

    # create checkpoint directory if none existed
    if os.path.isdir(args.checkpoint_path):
        print("Checkpointing directory {} exists".format(args.checkpoint_path))
    else:
        print("Creating Checkpointing directory {}".format(args.checkpoint_path))
        os.mkdir(args.checkpoint_path)
    
    
    # check if a checkpoint file exists within the checkpoint directory, if so, use it
    ckpt_file_path = None
    candidate_ckpt_file_path = args.checkpoint_path + '/last.ckpt'
    if os.path.exists(candidate_ckpt_file_path):
        ckpt_file_path = candidate_ckpt_file_path
    
    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.dist_backend, rank=host_rank, world_size=world_size)
        print(
            'Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
                args.dist_backend,
                dist.get_world_size()) + 'Current host rank is {}. Using cuda: {}. Number of gpus: {}'.format(
                dist.get_rank(), torch.cuda.is_available(), args.num_gpus))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device Type: {}".format(device))


    print("Load Time Series dataset from S3")
    data = pd.read_parquet("{}/{}".format(args.data_dir, args.data_filename))
    
    training_metadata = {}
    with open("{}/{}".format(args.data_dir, args.metadata_filename)) as json_file:
        training_metadata = json.load(json_file)


    print("creating dataloader")

    training = TimeSeriesDataSet(
        data[lambda x: x[training_metadata['time_idx']] <= training_metadata['training_cutoff']],
        time_idx= training_metadata['time_idx'],
        target= training_metadata['target'],
        group_ids= training_metadata['group_ids'],
        min_encoder_length= training_metadata['min_encoder_length'],  
        max_encoder_length= training_metadata['max_encoder_length'],
        min_prediction_length=training_metadata['min_prediction_length'],
        max_prediction_length=training_metadata['max_prediction_length'],
        static_categoricals=training_metadata['static_categoricals'],
        static_reals=training_metadata['static_reals'],
        time_varying_known_categoricals=training_metadata['time_varying_known_categoricals'],
        variable_groups= training_metadata['variable_groups'],  # group of categorical variables can be treated as one variable
        time_varying_known_reals= training_metadata['time_varying_known_reals'],
        time_varying_unknown_categoricals= training_metadata['time_varying_unknown_categoricals'],
        time_varying_unknown_reals= training_metadata['time_varying_unknown_reals'],
        target_normalizer=GroupNormalizer(
                groups= training_metadata['target_normalizer']['normalized_groups'], 
                transformation= training_metadata['target_normalizer']['normalization_transformation']
        ),  # use softplus and normalize by group
        add_relative_time_idx= training_metadata['add_relative_time_idx'],
        add_target_scales= training_metadata['add_target_scales'],
        add_encoder_length= training_metadata['add_encoder_length'],
        allow_missing_timesteps = training_metadata["allow_missing_timesteps"]
    )

    # create validation set (predict=True) which means to predict the last max_prediction_length points in time
    # for each series
    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

    # create dataloaders for model
    train_dataloader = training.to_dataloader(train=True, batch_size=minibatch_size, num_workers=os.cpu_count())
    val_dataloader = validation.to_dataloader(train=False, batch_size=minibatch_size * 10, num_workers=os.cpu_count())


    print("create model trainer")
    # configure network and trainer
    pl.seed_everything(42)
    
    
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=early_stopping_patience, verbose=False, mode="min")
    checkpoint_call_back = ModelCheckpoint(
            monitor='val_loss',
            dirpath = args.checkpoint_path,
            filename = '{epoch}-{val_loss:.2f}',
            save_last = True,
            auto_insert_metric_name = True
            
            
        )
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger(save_dir = "/lightning_logs", name = "TemporalFusionTransformer")  # logging results to a tensorboard

    
    trainer = pl.Trainer(
        max_epochs= num_epochs,
        accelerator= "auto",
        devices= "auto",
        weights_summary="top",
        gradient_clip_val= max_gradient_norm,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )
    
#     print("get GPU information")
#     num_GPU = 0
#     if torch.cuda.device_count() >= 1:
#         num_GPU = torch.cuda.device_count()
#         print("GPU count: {}".format(num_GPU))
    
#     trainer = pl.Trainer(
#         max_epochs=args.epochs,
#         gpus=num_GPU,
#         weights_summary="top",
#         gradient_clip_val=0.1,
# #         limit_train_batches=30,  # coment in for training, running valiation every 30 batches
#         # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
#         callbacks=[lr_logger, early_stop_callback, checkpoint_call_back],
#         logger=logger,
# #         enable_progress_bar=False
#     )

    print("create model from dataset")
    tft = TemporalFusionTransformer.from_dataset(
        training,
        # not meaningful for finding the learning rate but otherwise very important
        learning_rate= learning_rate,
        hidden_size= hidden_layer_size,        # most important hyperparameter apart from learning rate
        attention_head_size= num_heads,        # number of attention heads. Set to up to 4 for large datasets
        dropout= dropout_rate,                 # between 0.1 and 0.3 are good values
    #     hidden_continuous_size=8,              # set to <= hidden_size
        hidden_continuous_size=hidden_layer_size,  # set to <= hidden_size
        output_size=7,                         # 7 quantiles by default
        loss=QuantileLoss(),
        # reduce learning rate if no improvement in validation loss after x epochs
        reduce_on_plateau_patience= early_stopping_patience,
    )
#     tft = TemporalFusionTransformer.from_dataset(
#         training,
#         learning_rate=0.03,
#         hidden_size=16,
#         attention_head_size=1,
#         dropout=0.1,
#         hidden_continuous_size=8,
#         output_size=7,  # 7 quantiles by default
#         loss=QuantileLoss(),
#         log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
#         reduce_on_plateau_patience=4,
#     )
    print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")


    print("training model")
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path = ckpt_file_path
    )

    best_model_path = trainer.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    return _save_model(trainer, args.model_dir)


def _save_model(model_trainer, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, 'model_trainer.ckpt')
    # reference link: https://pytorch-lightning.readthedocs.io/en/stable/common/checkpointing_intermediate.html
    model_trainer.save_checkpoint(path)


def _save_checkpoint(model, optimizer, epoch, loss, args):
    pass

    
def _load_checkpoint(model, optimizer, args):
    pass

    
def model_fn(model_dir):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--dist_backend', type=str, default='gloo', help='distributed backend (default: gloo)')

    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument("--checkpoint-path",type=str,default="/opt/ml/checkpoints")
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    
    parser.add_argument('--data-filename', type=str, default="data.parquet")
    parser.add_argument('--metadata-filename', type=str, default="metadata.json")
    
    
    
    parser.add_argument('--max-prediction-length', type=int, default= 24)
    parser.add_argument('--max-encoder-length', type=int, default=24 * 7)
    parser.add_argument('--num-epochs', type=int, default=2, metavar='E',
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('--early-stopping-patience', type=int, default= 5)
    parser.add_argument('--dropout-rate', type=float, default=0.1)
    parser.add_argument('--hidden-layer-size', type=int, default=160)
    parser.add_argument('--learning-rate', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--minibatch-size', type=int, default=4, metavar='BS',
                        help='batch size (default: 4)')
    parser.add_argument('--max-gradient-norm', type=float, default=0.01)
    parser.add_argument('--num-heads', type=int, default=4)
    
    if int(os.environ['SM_NUM_GPUS']) > 0:
        parser.add_argument('--multiprocessing-workers', type=int, default=os.environ['SM_NUM_GPUS'])
    else:
        parser.add_argument('--multiprocessing-workers', type=int, default=os.environ['SM_NUM_CPUS'])

    
    
    
    _train(parser.parse_args())