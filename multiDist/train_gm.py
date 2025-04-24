import argparse
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
import wandb
import yaml
from emir.estimators.knife import KNIFE
from emir.estimators.knife_estimator import KNIFEArgs
from multiDist.trainer.trainer_gm import *
from multiDist.utils.parser import update_grouped_models, get_pretraining_args
from multiDist.utils.data_multifiles import get_embedding_loader
from multiDist.utils.model import get_student_model
from multiDist.utils.embedder_info import get_embedder_size


def get_parser():
    parser = get_pretraining_args()

    # model parameters

    parser.add_argument("--knifes-config", type=str, default="hp/knifes.yaml")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--use-teacher-bn", action="store_true")
    parser.set_defaults(use_teacher_bn=True)

    return parser


def main(args):
    '''Get All Embeddings Datasets '''

    train_loader, valid_loader, embs_dim = get_embedding_loader(args)

    '''Create Teacher KNIFE'''
    
    if os.path.exists(args.knifes_config):
        with open(args.knifes_config, "r") as f:
            knifes_config = yaml.safe_load(f)
            knifes_config = KNIFEArgs(**knifes_config)
    else:
        knifes_config = KNIFEArgs(device=args.device)
        os.makedirs(os.path.dirname(args.knifes_config), exist_ok=True)
        with open(args.knifes_config, "w") as f:
            yaml.dump(knifes_config.__dict__, f)

    '''Extended to Have a Knife for Each Embedder of Each Modality'''
    knifes = {}
    for emb_key in embs_dim.keys():
        knife = []
        for emb_dm in embs_dim[emb_key]:
            knife.append(KNIFE(
                    args=knifes_config,
                    zc_dim=args.out_dim,
                    zd_dim=emb_dm,
                ).kernel_cond)
        knife = torch.nn.ModuleList(knife)
        knifes[emb_key] = knife


    model = get_student_model(args)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    criterion = torch.nn.L1Loss()
    scheduler = None  # torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    # optimizer, T_0=(args.num_epochs * 4) // 10, eta_min=args.lr / 100, T_mult=1
    # )
    embedder_name_list = {}
    if "vision" in args.modalities_to_simulate:
        embedder_name_list["vision"] = args.vision_embedders_to_simulate
    if "text" in args.modalities_to_simulate:
        embedder_name_list["text"] = args.text_embedders_to_simulate
    if "molecular" in args.modalities_to_simulate:
        embedder_name_list["molecular"] = args.molecular_embedders_to_simulate


    trainer = TrainerGM(
        model,
        knifes,
        optimizer,
        criterion,
        scheduler=scheduler,
        device=args.device,
        batch_size=args.batch_size,
        wandb=args.wandb,
        embedder_name_list=embedder_name_list,
        out_dir=args.out_dir,
        mods = args.modalities_to_simulate
    )


    trainer.train(
        train_loader,
        valid_loader,
        args.num_epochs,
        args.log_interval,
    )

    




if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    args.molecular_embedders_to_simulate = update_grouped_models(args.molecular_embedders_to_simulate)
    args.embedders_to_simulate = {"text": args.text_embedders_to_simulate, \
                                  "molecular": args.molecular_embedders_to_simulate, \
                                    "vision": args.vision_embedders_to_simulate}
    args.student_emb_size = {}
    args_dict = vars(args)
    for mod in args.modalities_to_simulate:
        args.student_emb_size[mod] = get_embedder_size(args_dict[mod + "_student"])
    del args_dict
    
    if args.wandb:
        wandb.init(
            project="multi-mod-distill-test",
            allow_val_change=True,
        )

        if not wandb.run.name is None:
            args.out_dir = os.path.join(args.out_dir, wandb.run.name)
        print(args.out_dir)

        wandb.config.update(args)

        wandb.define_metric("train_loss", step_metric="epoch")
        wandb.define_metric("eval_loss", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")
        '''Define the metrice for each embedder of each modality'''
        for mod in args.modalities_to_simulate:
            for embedder in args.embedders_to_simulate[mod]:
                wandb.define_metric(f"train_loss_{embedder}", step_metric="epoch")
                wandb.define_metric(f"test_loss_{embedder}", step_metric="epoch")
                
    os.makedirs(args.out_dir, exist_ok=True)
    
    main(args)
    