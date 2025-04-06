from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from datamodule import PAFDatamodule
from esm import ProteinClassifier

if __name__ == "__main__":
    datamodule = PAFDatamodule("../datafiles", batch_size=16)
    model = ProteinClassifier(n_classes=25)
    
    # early stop and callbacks
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, mode="min"),
        ModelCheckpoint(
            dirpath="checkpoints",
            filename="best_model",
            monitor="val_loss",
            mode="min",
            save_top_k=1
        )
    ]
    
    trainer = Trainer(
        max_epochs=10,
        callbacks=callbacks,
        enable_progress_bar=True
    )
    trainer.fit(model=model, datamodule=datamodule)