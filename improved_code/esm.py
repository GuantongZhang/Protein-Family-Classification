import os
import torch
import torchmetrics
from lightning import LightningModule
from torch import nn
from torch.optim import AdamW
from transformers import AutoTokenizer, EsmModel

class ProteinClassifier(LightningModule):
    def __init__(self, n_classes=25):
        super().__init__()
        self.save_hyperparameters()

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.embedder = EsmModel.from_pretrained("facebook/esm2_t6_8M_UR50D")
        
        self.classifier = nn.Linear(320, n_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=n_classes)

        self.training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
    
    def forward(self, x):
        '''
        Forward pass for protein sequence classification
        Args:
            x: List of protein sequence strings
        Returns:
            logits: Raw prediction scores for each class (before softmax)
        '''
        # Tokenize input sequences
        encoded = self.tokenizer(
            x,
            add_special_tokens=True,
            padding="longest",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        # Move tensors to model's device (GPU/CPU)
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # Get all token embeddings
        outputs = self.embedder(input_ids=input_ids, attention_mask=attention_mask)
        all_token_embeddings = outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)
        
        # Masked mean pooling (exclude padding tokens)
        masked_embeddings = all_token_embeddings * attention_mask.unsqueeze(-1)  # Zero-out pad
        sum_embeddings = masked_embeddings.sum(dim=1)  # (batch_size, hidden_size)
        valid_lengths = attention_mask.sum(dim=1).unsqueeze(-1)  # (batch_size, 1)
        embeddings = sum_embeddings / valid_lengths  # Normalize by actual sequence length
        
        # Final classification layer
        logits = self.classifier(embeddings)
        return logits
    

    def training_step(self, batch, batch_idx):
        '''
        calculate output --> loss --> training accuracy and save to self.log
        return loss
        '''
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(preds, y)
        
        # Calculate training accuracy
        #preds = torch.argmax(logits, dim=1)
        #self.train_accuracy.update(preds, y)
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        
        return loss

    def on_train_epoch_end(self):
        # Save training metrics
        self.training_history['train_loss'].append(self.trainer.callback_metrics['train_loss'].item())
        self.training_history['train_acc'].append(self.trainer.callback_metrics['train_acc'].item())
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)
        self.val_f1.update(preds, y)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        # Save validation metrics
        self.training_history['val_loss'].append(self.trainer.callback_metrics['val_loss'].item())
        self.training_history['val_acc'].append(self.trainer.callback_metrics['val_acc'].item())
        self.training_history['val_f1'].append(self.trainer.callback_metrics['val_f1'].item())

    def on_fit_end(self):
        os.makedirs("checkpoints", exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'training_history': self.training_history
        }, "checkpoints/protein_classifier.pt")

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)