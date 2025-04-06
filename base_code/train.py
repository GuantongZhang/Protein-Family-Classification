from lightning import Trainer
from datamodule import PAFDatamodule
from prot_bert import ProteinClassifier

if __name__ == "__main__":
    datamodule = PAFDatamodule("../datafiles", batch_size=16)
    model = ProteinClassifier(n_classes=25)
    trainer = Trainer(max_epochs=3)
    #trainer.fit(model=model, datamodule=datamodule)
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path="lightning_logs/version_30250421/checkpoints/epoch=0-step=4163.ckpt"
    )