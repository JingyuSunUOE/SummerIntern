import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import torch
import torch.nn as nn
import torch.nn.functional as F

class UQSamplingPredictorLitModelWrapper(pl.LightningModule):
    def __init__(self, model, loss, sampler, is_uq_model=False):
        super().__init__()
        """
        pytorch lightning module for doing UQ sampling
        """
        self.model = model
        self.loss = loss
        self.is_uq_model = False
        self.sampler = sampler

        
    def forward(self, x, **kwargs):
        return self.sampler(x, self.model, **kwargs)
    
    
    def training_step(self, batch, batch_idx):
        raise ValueError("The UQ sampler wrapper is for inference on trained models only, not for training")
    
    def validation_step(self, batch, batch_idx):
        """
        note: call trainer.validate() automatically loads the best checkpoint if checkpointing was enabled during fitting
        well yes I want to enable checkpointing but will deal with that later.
        also it does stuff like model.eval() and torch.no_grad() automatically which is nice.
        I will need a custom eval thing to do my dropout estimation but can solve that later too.
        """
        if self.is_uq_model:
            self.model.set_applyfunc(True) # now we let the sampler deal with the stochasticity
        
        X, y = batch
        samples, y_hat = self(X)
        val_loss = self.loss(y_hat, y)
        
        # if self.logging_metrics != None:
        #     for i, lm in enumerate(self.logging_metrics):
        #         value = lm[1](y_hat, y)
        #         self.log(f"metric: {i+1}", value, prog_bar=True)
        self.log("val_loss", val_loss)
        
    def test_step(self, batch, batch_idx):
        """
        we would need to directly call this function using the trainer
        """
        
        if self.is_uq_model:
            self.model.set_applyfunc(True) # now we let the sampler deal with the stochasticity
        
        X, y = batch
        samples, y_hat = self(X)
        test_loss = self.loss(y_hat, y)
        self.log("test_loss", test_loss)
        
    def predict_step(self, batch, batch_idx):
        """
        returns both the samples and the maximum likelihood prediction
        as well as the targets
        """
        
        if self.is_uq_model:
            self.model.set_applyfunc(True) # now we let the sampler deal with the stochasticity
        
        X, y = batch
        samples, mle_est = self(X)
        return samples, mle_est, X, y