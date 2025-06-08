import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, Precision, Recall, F1Score
from model import PolypDetectionModel

class PolypDetectionLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super().__init__()
        self.model = PolypDetectionModel()
        self.learning_rate = learning_rate
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Metrics
        self.train_accuracy = Accuracy(task='binary')
        self.val_accuracy = Accuracy(task='binary')
        self.test_accuracy = Accuracy(task='binary')
        
        self.train_f1 = F1Score(task='binary')
        self.val_f1 = F1Score(task='binary')
        self.test_f1 = F1Score(task='binary')
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels.float().unsqueeze(1))
        
        # Log metrics
        self.train_accuracy(outputs, labels)
        self.train_f1(outputs, labels)
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy, prog_bar=True)
        self.log('train_f1', self.train_f1, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels.float().unsqueeze(1))
        
        # Log metrics
        self.val_accuracy(outputs, labels)
        self.val_f1(outputs, labels)
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy, prog_bar=True)
        self.log('val_f1', self.val_f1, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels.float().unsqueeze(1))
        
        # Log metrics
        self.test_accuracy(outputs, labels)
        self.test_f1(outputs, labels)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_accuracy, prog_bar=True)
        self.log('test_f1', self.test_f1, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.1, patience=5, verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        } 