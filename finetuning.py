import argparse
import torch
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as L
from torchmetrics.functional import auroc
from utils.ModelTesting import TunnerModel
from Data.DataClass import *
import pandas as pd
import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import WandbLogger
import wandb
import logging
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from utils.teachers_dict import teachers_dict
from Data.dataset_dict import datasets_dict
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=str, default="resnet18")
    parser.add_argument("--type", type=str, default="")
    parser.add_argument("--save-dir", type=str, default="./output_student")
    return parser

def save_predictions(model, output_fname, num_classes):
    prds = torch.cat(model.predictions, dim=0)
    trgs = torch.cat(model.targets, dim=0)

    auc = auroc(prds, trgs, num_classes=num_classes, average='macro', task='multiclass')

    print('AUROC (test)')
    print(auc)

    cols_names = ['class_' + str(i) for i in range(0, num_classes)]

    df = pd.DataFrame(data=prds.cpu().numpy(), columns=cols_names)    
    df['target'] = trgs.cpu().numpy()
    df.to_csv(output_fname, index=False)
    l = []
    for i in range(num_classes):
        l.append(df['class_'+ str(i)])
    preds = np.stack(l).transpose()
    targets = np.array(df['target'])
    print("balanced accuracy, F1 score:")
    print(balanced_accuracy_score(targets, preds.argmax(1)), f1_score(targets, preds.argmax(1), average='micro'), accuracy_score(targets, preds.argmax(1)))

L.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "gpu" if torch.cuda.is_available() else "cpu"
num_workers = 30
batch_size = 64

def finetune(data, model_, output_name, num_classes):
    model = TunnerModel(model=model_, num_classes = num_classes)
    wandb.init(project="distill_compare_"+output_name, reinit=True)
    output_dir = os.path.join(output_base_dir,output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    monitor="val_acc"
    mode='max'
    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode=mode)
    early_stop_callback = EarlyStopping(
            monitor="val_acc",
            min_delta=0.001,
            patience=20,
            verbose=False,
            mode="max",
        )
    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode=mode)
    wandb_logger = WandbLogger(log_model=False)
    trainer = pl.Trainer(
            callbacks=[checkpoint_callback, early_stop_callback],#
            log_every_n_steps=1,
            max_epochs=50,
            accelerator=device,
            devices=1,
            logger=[wandb_logger, TensorBoardLogger(output_base_dir, name=output_name), CSVLogger("logs_teacher", name=output_name)],#
        )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)
    print(trainer.checkpoint_callback.best_model_path)
    model = TunnerModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes = num_classes, model = model_)
    print(trainer.test(model=model, datamodule=data))
    save_predictions(model, os.path.join(output_dir, 'predictions.csv'), num_classes)

parser = get_parser()
args = parser.parse_args()
student = args.student
output_base_dir = args.save_dir
for dataset in datasets_dict.keys():
    print(student, dataset, output_base_dir)
    data = data_module_dict[dataset](batch_size, num_workers)
    model_ = teachers_dict[student](pretrained=True)
    if args.type != "teacher":
        PATH = "./results/"+student+"_student_" + args.type + "/best_model.pth"
        checkpoint = torch.load(PATH)
        state_dict = {k.replace(" ",""): v for k, v in checkpoint.items()}
        model_.load_state_dict(state_dict)
    output_name = student + "_" + dataset + "_finetuning_" + args.type
    print(output_name)
    finetune(data, model_, output_name, data_class_dict[dataset])