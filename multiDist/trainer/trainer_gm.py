import os
import time
import itertools
import torch
from tqdm import tqdm
import torch.nn as nn
from multiDist.trainer.tracing_decorator import tracing_decorator

class TrainerGM:
    def __init__(
        self,
        model,
        knifes,
        optimizer,
        criterion,
        device,
        batch_size,
        scheduler=None,
        wandb=False,
        embedder_name_list=None,
        out_dir=None,
        mods = None,
    ):
        self.model = model
        self.knifes = knifes
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.wandb = wandb
        self.embedder_name_list = embedder_name_list
        self.batch_size = batch_size
        self.knife_optimizer = torch.optim.AdamW(itertools.chain.from_iterable(model.parameters() for model in knifes.values()), lr=1e-3)
        self.out_dir = out_dir
        self.mods = mods

    @tracing_decorator("knife")
    def get_knife_loss(self, embeddings, batch, loss_per_embedder=None):
        loss = 0
        for mod_idx in range(len(self.mods)):
            mod = self.mods[mod_idx]
            losses = [torch.tensor(0.0)]*len(self.knifes[mod])
            indexes = [i for i in range(len(batch[mod + "_emb"])) if batch[mod + "_emb"][i] is not None]
            if len(indexes) == 0:
                continue
            for i in range(len(self.knifes[mod])):
                data = [batch[mod + "_emb"][j][i] for j in indexes]
                #print(mod, i, torch.tensor(data).to(self.device, non_blocking=True).shape, embeddings[indexes].shape)
                losses[i] = losses[i] + self.knifes[mod][i](embeddings[indexes], torch.tensor(data).to(self.device, non_blocking=True))

            loss += sum(losses)
            if loss_per_embedder is not None:
                for i, l in enumerate(losses):
                    loss_per_embedder[self.embedder_name_list[mod][i]] += l

        return loss


    def train(
        self,
        train_loader,
        valid_loader,
        num_epochs,
        log_interval,
    ):
        min_eval_loss = float("inf")

        for epoch in range(num_epochs):
            train_loss, train_loss_per_embedder = self.train_epoch(train_loader, epoch)
            dict_to_log = {
                "train_loss": train_loss,
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            if epoch % log_interval == 0:
                
                eval_loss, test_loss_per_embedder = self.eval(valid_loader, epoch)
                print(f"Epoch {epoch}, Loss: {train_loss}, Eval Loss: {eval_loss}")
                dict_to_log["eval_loss"] = eval_loss
                if eval_loss < min_eval_loss:
                    min_eval_loss = eval_loss
                    torch.save(
                        self.model.state_dict(),
                        os.path.join(self.out_dir, "best_model.pth"),
                    )
            if self.wandb:
                import wandb

                for name, loss in train_loss_per_embedder.items():
                    dict_to_log[f"train_loss_{name}"] = loss
                for name, loss in test_loss_per_embedder.items():
                    dict_to_log[f"eval_loss_{name}"] = loss
                wandb.log(dict_to_log)
            if self.scheduler is not None:
                self.scheduler.step()
        return
    
    
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        train_loss = 0
        train_loss_per_embedder = {name: 0 for mod in self.embedder_name_list.keys() for name in self.embedder_name_list[mod]}
        for batch_idx, batch in enumerate(
            tqdm(
                train_loader,
                desc=f"Training || epoch {epoch} ||  ",
                total=len(train_loader),
            ),
        ):
            self.optimizer.zero_grad()
            self.knife_optimizer.zero_grad()


            loss = self.get_loss(
                batch,
                backward=True,
                loss_per_embedder=train_loss_per_embedder,
            )
            train_loss += loss
        for name in train_loss_per_embedder.keys():
            train_loss_per_embedder[name] = train_loss_per_embedder[name].item() / len(
                train_loader
            )
        return train_loss.item() / len(train_loader), train_loss_per_embedder
    

    @torch.no_grad()
    def eval(self, valid_loader, epoch):
        self.model.eval()
        eval_loss = 0
        test_loss_per_embedder = {name: 0 for mod in self.embedder_name_list.keys() for name in self.embedder_name_list[mod]}
        for batch_idx, batch in enumerate(
            tqdm(
                valid_loader,
                desc=f"Eval || epoch {epoch} ||  ",
                total=len(valid_loader),
            )
        ):
            l, embs = self.get_loss(
                batch,
                backward=False,
                return_embs=True,
                loss_per_embedder=test_loss_per_embedder,
            )
            eval_loss += l

        for name in test_loss_per_embedder.keys():
            test_loss_per_embedder[name] = test_loss_per_embedder[name].item() / len(
                valid_loader
            )
        return eval_loss.item() / len(valid_loader), test_loss_per_embedder
    
    def get_loss(
        self,
        batch,
        backward=True,
        loss_per_embedder=None,
        return_embs=False,
    ):
        ''''''
        
        embeddings = self.model(batch)
        loss = self.get_knife_loss(
            embeddings, batch, loss_per_embedder=loss_per_embedder
        )
        if backward:
            loss.backward()
            self.optimizer.step()
            self.knife_optimizer.step()
            self.optimizer.zero_grad()
            self.knife_optimizer.zero_grad()
        if return_embs:
            return loss, embeddings
        return loss
