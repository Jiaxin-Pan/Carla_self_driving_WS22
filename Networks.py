"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import random
import torchvision.models as models

class DummySteerModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.prediction = random.random()
    def forward(self, x):
        return random.uniform(-1,1)   #output a random number between -1 and 1

    
class SteerNN(pl.LightningModule):

    def __init__(self,hparams1=None, train_set=None, val_set=None):
        super().__init__()
        self.hparams1 = hparams1
        self.train_set = train_set
        self.val_set = val_set

        
        self.features = models.alexnet(pretrained=True).features   #output: N*256*17*24
        self.predict = nn.Sequential(
            nn.Flatten(),                  #keep the first dim N, the others will be flattened
            nn.Linear(256*17*24, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x):

        x = x.view(-1, 3, 600, 800)
        x = self.features(x)
        #print(x.shape)
        x = self.predict(x)


        return x
    
    def general_step(self, batch, batch_idx, mode):
        image, steer = batch["image"], batch["steer"]
        #flattened_images = images.view(images.shape[0], -1)
        #keypoints = torch.squeeze(keypoints)

        # forward pass
        #out = self.forward(image).view(-1,15,2)
        #out = torch.squeeze(out)
        out = self.forward(image)
        out = torch.squeeze(out)
        out = out.double()
        steer = steer.double()
        # loss
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(out, steer)

        return loss
    
    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        return avg_loss


    def training_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "train")
        print("training loss", loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        loss = self.general_step(batch, batch_idx, "val")
        print("validation loss", loss)
        return {'val_loss': loss}
    

    def validation_end(self, outputs):
        avg_loss = self.general_end(outputs, "val")
        print(avg_loss)
        return {'val_loss': avg_loss}

    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, shuffle=True, batch_size=self.hparams1['batch_size'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.hparams1['batch_size'])
    
    def configure_optimizers(self):

        optim = torch.optim.Adam(self.predict.parameters(), self.hparams1["learning_rate"])
        
        return optim
