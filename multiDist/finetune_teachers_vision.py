import argparse
import json
import torch
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as L
from torchmetrics.functional import auroc
from multiDist.utils.vision_utils import TunnerModel, TunnerModelFS, datasets_dict
#from Data.DataClass import *
from multiDist.utils.vision_utils import data_module_dict, data_class_dict
import pandas as pd
import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import RichProgressBar
import wandb
import logging
from sklearn.metrics import top_k_accuracy_score, balanced_accuracy_score, accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from multiDist.utils.vision_utils import teachers_dict, EmbedderFromTorchvision, embedder_size, teachers_dict_vit_all, EmbedderFromViT, ModelImageTransform


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
    return accuracy_score(targets, preds.argmax(1))

L.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "gpu" if torch.cuda.is_available() else "cpu"
num_workers = 8
output_base_dir = 'output_teacher'

def finetune(args, data, model_, output_name, num_classes, num_epoch, dim, few_shot, fc_layers_config):
    model = TunnerModel(
        lr = args.lr, 
        weight_decay = args.weight_decay, 
        model=model_, 
        num_classes = num_classes, 
        output_dim = dim,
        fc_layers_config = fc_layers_config, 
        scheduler = args.scheduler,
        norm=args.norm
        )
    output_dir = os.path.join(output_base_dir,output_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    monitor="val_acc"
    mode='max'
    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode=mode)
    early_stop_callback = EarlyStopping(
            monitor="val_acc",
            min_delta=0.001,
            patience=args.patience,
            verbose=False,
            mode="max",
            
        )
    checkpoint_callback = ModelCheckpoint(monitor=monitor, mode=mode)
    logger = [CSVLogger("logs_teacher", name=output_name)]
    if args.wandb:
        wandb_logger = WandbLogger(log_model=False)
        logger = [wandb_logger, CSVLogger("logs_teacher", name=output_name)]
    trainer = pl.Trainer(
            callbacks=[checkpoint_callback, early_stop_callback, RichProgressBar()],#
            log_every_n_steps=1,
            max_epochs=num_epoch,
            accelerator=device,
            devices=1,
            logger=logger,#
        )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)
    print(trainer.checkpoint_callback.best_model_path)
    model = TunnerModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes = num_classes, model = model_, fc_layers_config = fc_layers_config, output_dim = dim, norm=args.norm)
    model.eval()
    print(trainer.test(model=model, datamodule=data))
    acc = save_predictions(model, os.path.join(output_dir, 'predictions.csv'), num_classes)
    if args.wandb:
        wandb_logger.log_metrics({"test_acc": acc})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
        
    parser.add_argument("--teachers", nargs="+", type=str, default=list(teachers_dict.keys()))
    parser.add_argument("--datasets", nargs="+", type=str, default=list(datasets_dict.keys()))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--student", action="store_true")
    parser.add_argument("--fewshow", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--scheduler", type=bool, default=False)
    parser.add_argument("--mixup", type=bool, default=False)
    parser.add_argument("--shot", type=int, default=10)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--type", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--augmentations", type=str, default=json.dumps({"ColorJitter": {"brightness" : 0.2}, "RandomHorizontalFlip": True}), help="JSON string for augmentations.")
    parser.add_argument("--fc_layers", type=int, default=0)
    parser.add_argument("--norm", type=str, default="")
    args = parser.parse_args()
    args.augmentations = args.augmentations.replace("'", '"').replace("True", "true").replace("False", "false")


    args.augmentations = json.loads(args.augmentations)

    for teacher in args.teachers:
        for dataset in args.datasets:
            transform = None
            if teacher in teachers_dict_vit_all.keys():
                model_ = EmbedderFromViT(teacher)
                transform = ModelImageTransform(teacher)
            else:
                model_ = EmbedderFromTorchvision(teacher, pretrained = True)
                #embedder_size[teacher] = 1000
            print(".......................................", args.student)
            if args.student:
                print("here")
                if teacher in teachers_dict_vit_all.keys():
                    model_ = EmbedderFromViT(teacher)
                    transform = ModelImageTransform(teacher)
                else:
                    model_ = teachers_dict[teacher]
                    embedder_size[teacher] = 1000
                checkpoint = torch.load(args.checkpoint)
                model_.load_state_dict(checkpoint)
            data = data_module_dict[dataset](args.batch_size, num_workers, transform = transform, augmentations=args.augmentations)#, mixup = args.mixup
            output_name = teacher + "_" + dataset + "_teacher_"+args.type+"finetuning"
            if args.student:
                output_name = teacher + "_" + dataset + "_student_"+args.type+"finetuning"
            print(output_name)
            if args.wandb:
                wandb.init(project="Distill-Downstream-Vision-Swin", reinit=True)
                wandb.config.update(args)
            #wandb.config["datasets"] = dataset

            finetune(args, data, model_, output_name, data_class_dict[dataset], args.epochs, embedder_size[teacher], args.fewshow, fc_layers_config = args.fc_layers)#