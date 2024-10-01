import lightning as pl
import torch
from torch.optim import AdamW
from torch import nn


class DistilledEmbedderPLModel(pl.LightningModule):
    def __init__(self, model, teachers_kernels, lr, train_normalized=True):
        super().__init__()
        self.model = model
        self.teachers_count = len(teachers_kernels)
        self.teachers_kernels = nn.ModuleList(teachers_kernels)
        self.lr = lr

        self.forward = self.model.forward

    def training_step(self, batch, batch_idx):
        inputs, targets = batch  # img, List[Tensor]

        negative_log_likelihood = []
        teachers_indices = [i for i in range(self.teachers_count)]
        if inputs.shape[1] < 3:#?
                inputs = torch.stack((inputs,inputs,inputs), dim = 1).squeeze(2)
        for k, teacher_idx in enumerate(teachers_indices):
            output = self.model(inputs)
            #output = output.last_hidden_state[:, 0]  # get the logits forward pass, returns a tensor (batch_size, hidden_size)
            # get the teacher kernel
            teacher_kernel = self.teachers_kernels[teacher_idx]

            nll = -teacher_kernel.logpdf(output, targets[k]).mean()
            negative_log_likelihood.append(nll * (1 / targets[k].shape[-1]))

            self.log(
                f"train_nll_{teacher_idx}",
                nll,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=inputs.shape[0],
            )

        negative_log_likelihood = sum(negative_log_likelihood)

        self.log(
            "train_nll",
            negative_log_likelihood,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=sum(inputs.shape[0] for k in range(len(inputs))),
        )

        return negative_log_likelihood

    def validation_step(self, batch, batch_idx):
        inputs, targets = (
            batch  # List[Tensor], Dict [str, Tensor], List[int]
        )

        negative_log_likelihood = 0
        teachers_indices = [i for i in range(self.teachers_count)]
        if inputs.shape[1] < 3:#?
                inputs = torch.stack((inputs,inputs,inputs), dim = 1).squeeze(2)
        for k, teacher_idx in enumerate(teachers_indices):
            output = self.model(inputs)
            #output = output.logits

            teacher_kernel = self.teachers_kernels[teacher_idx]
            nll = -teacher_kernel.logpdf(output, targets[k]).mean()
            negative_log_likelihood += nll

            self.log(
                f"val_nll_{teacher_idx}",
                nll,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=inputs.shape[0],
            )

        self.log(
            "val_nll",
            negative_log_likelihood,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=sum(inputs.shape[0] for k in range(len(inputs))),
        )
        return negative_log_likelihood

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)

        return optimizer

class DistilledEmbedderPLModelAlignedInputs(pl.LightningModule):
    def __init__(self, model, teachers_kernels, lr=1e-3, train_normalized=True):
        super().__init__()
        self.model = model

        self.teachers_kernels = nn.ModuleList(teachers_kernels)
        self.lr = lr
        self.train_normalized = train_normalized

        self.save_hyperparameters("lr")
        self.save_hyperparameters("train_normalized")

    def forward(self, x):
        return self.model(**x)

    def training_step(self, batch, batch_idx):
        # List[Tensor], hf model inputs dict, Tensor : (n_teachers, batch_size)
        targets, inputs, mask = batch

        output = self.model(**inputs)
        output = output.last_hidden_state[:, 0]

        negative_log_likelihood = []
        for k, teacher_kernel in enumerate(self.teachers_kernels):
            selected = torch.where(mask[k])
            output_k = output[selected]
            target_k = targets[k][selected]
            if output_k.shape[0] == 0:
                continue
            nll = -teacher_kernel.logpdf(output_k, target_k).mean()
            negative_log_likelihood.append((nll, targets[k].shape[-1]))

            self.log(
                f"train_nll_{k}",
                nll,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                batch_size=output_k.shape[0],
            )

        negative_log_likelihood_normalized = sum(
            nll / dim for nll, dim in negative_log_likelihood
        )
        negative_log_likelihood = sum(nll for nll, _ in negative_log_likelihood)

        self.log(
            "train_nll",
            negative_log_likelihood,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=sum(mask[k].sum() for k in range(len(mask))),
        )

        self.log(
            "train_nll_normalized",
            negative_log_likelihood_normalized,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=sum(mask[k].sum() for k in range(len(mask))),
        )

        if self.train_normalized:
            return negative_log_likelihood_normalized
        else:
            return negative_log_likelihood

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)

        return optimizer