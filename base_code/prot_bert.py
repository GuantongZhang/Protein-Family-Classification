import os
import torch
import torchmetrics
from lightning import LightningModule
from torch import nn
from torch.optim import AdamW
from transformers import BertTokenizer, BertModel

class ProteinClassifier(LightningModule):
    def __init__(self, n_classes=25):
        super().__init__()
        self.save_hyperparameters()
        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        self.embedder = BertModel.from_pretrained("Rostlab/prot_bert")
        self.classifier = nn.Linear(1024, n_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)
        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=n_classes)
        self.val_f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=n_classes)
    
    def forward(self, x):
        '''
        Forward pass for protein sequence classification
        Args:
            x: List of protein sequence strings
        Returns:
            logits: Raw prediction scores for each class (before softmax)
        '''
        # Tokenize input sequences with BERT tokenizer
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
        
        # Get all token embeddings from BERT
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
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy.update(preds, y)
        
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        
        return loss


    def validation_step(self, batch, batch_idx):
        '''
        make predictions and calculate validation accuracy/F1 score and save to self.log
        '''
        x, y = batch
        logits = self(x)
        
        # Calculate validation metrics
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)
        self.val_f1.update(preds, y)
        
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

    def on_fit_end(self):
        # save finel models
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(self.state_dict(), "checkpoints/protein_classifier.pt")

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
