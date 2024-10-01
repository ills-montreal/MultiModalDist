import torch
import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as L
from torchmetrics.functional import auroc
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics.classification import BinaryAccuracy
class TunnerModel(L.LightningModule):
    def __init__(
        self,
        model = models.resnet18(pretrained=True),
        output_dim = 1000,
        lr=0.001, momentum=0.9, nesterov = True, weight_decay = 0.0001, batch_size = 64, num_classes = 100,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.model = model
        for param in self.model.parameters():
            param.requires_grad = False
        self.linear1 = nn.Linear(output_dim, output_dim)
        self.activation = nn.LeakyReLU()
        self.fc = nn.Linear(output_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov
        self.weight_decay = weight_decay
        if num_classes == 2:
            self.metric = BinaryAccuracy()
        else:
            self.metric = MulticlassAccuracy(num_classes=num_classes)

        self.predictions = []
        self.targets = []

        self.train_step_preds = []
        self.train_step_trgts = []
        self.val_step_preds = []
        self.val_step_trgts = []
        self.train_loss = []
        self.val_loss = []
        
    def forward(self, x):
        x = self.model(x)
        x = self.linear1(x)
        x = self.activation(x)
        out = self.fc(x)
        return out
    
    def compute_loss(self, y, yp):
        return F.cross_entropy(y, yp)
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr = self.lr, momentum = self.momentum, nesterov = self.nesterov, weight_decay = self.weight_decay)
        #scheduler = CosineAnnealingLR(optimizer, T_max = 12500)
        return {"optimizer": optimizer}#, "lr_scheduler": scheduler
    
    def process_batch(self, batch):
        img = batch[0].to(self.device)
        lab = batch[1].to(self.device)
        if img.shape[1] < 3:#?
            img = torch.stack((img,img,img), dim = 1).squeeze(2)
        out = self.forward(img)
        prd = torch.softmax(out, dim=1)
        loss = self.compute_loss(prd, lab)
        return loss, prd, lab

    def training_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.train_step_preds.append(prd)
        self.train_step_trgts.append(lab)
        self.log('train_loss', loss, 
            on_step=True,
            on_epoch=False,
            prog_bar=True, batch_size=self.batch_size)        
        '''batch_ratio = len(np.where(lab.cpu().numpy() == 1)[0]) / len(np.where(lab.cpu().numpy() == 0)[0])
        self.log('batch_ratio', batch_ratio, batch_size=self.batch_size)                        
        grid = torchvision.utils.make_grid(batch['image'][0:4, ...], nrow=2, normalize=True)
        self.logger.experiment.add_image('images', grid, self.global_step)'''
        return loss

    def on_train_epoch_end(self):
        all_preds = torch.cat(self.train_step_preds, dim=0)
        all_trgts = torch.cat(self.train_step_trgts, dim=0)
        auc = auroc(all_preds, all_trgts, num_classes=self.num_classes, average='macro', task='multiclass')
        acc = self.metric(all_preds.argmax(1), all_trgts)
        self.log('train_auc', auc, batch_size=len(all_preds))
        self.log('train_acc', acc, batch_size=len(all_preds))
        self.train_step_preds.clear()
        self.train_step_trgts.clear()

    def validation_step(self, batch, batch_idx):
        loss, prd, lab = self.process_batch(batch)
        self.val_step_preds.append(prd)
        self.val_step_trgts.append(lab)
        self.log('val_loss', loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True, batch_size=self.batch_size)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.val_step_preds, dim=0)
        all_trgts = torch.cat(self.val_step_trgts, dim=0)
        auc = auroc(all_preds, all_trgts, num_classes=self.num_classes, average='macro', task='multiclass')
        acc = self.metric(all_preds.argmax(1), all_trgts)
        self.log('val_auc', auc, batch_size=len(all_preds))
        self.log('val_acc', acc, batch_size=len(all_preds))
        self.val_step_preds.clear()
        self.val_step_trgts.clear()

    def on_test_start(self):
        self.predictions = []
        self.targets = []

    def test_step(self, batch, batch_idx):
        _, prd, lab = self.process_batch(batch)        
        self.predictions.append(prd)
        self.targets.append(lab.squeeze())
