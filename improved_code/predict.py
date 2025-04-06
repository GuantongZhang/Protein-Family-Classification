import os
import torch
import pandas as pd
from datamodule import PAFDatamodule
from esm import ProteinClassifier
from lightning.pytorch.utilities.migration import pl_legacy_patch

class Predictor:
    def __init__(self, checkpoint_path):
        # Initialize model
        self.model = ProteinClassifier(n_classes=25)
        self._load_weights(checkpoint_path)
        self.model.eval()
        
        # Initialize datamodule
        self.datamodule = PAFDatamodule(root_path="../datafiles", batch_size=8)
        
        # Setup tokenizer
        self.tokenizer = self.model.tokenizer

    def _load_weights(self, checkpoint_path):
        """Handle different checkpoint formats"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

    def predict(self):
        """Run prediction on test data"""
        # Get test data
        test_data = self.datamodule.get_dataset("test", with_target=False)
        #test_data = test_data[:10] ###

        # Process in batches
        predictions = []
        batch_size = self.datamodule.batch_size
        
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            with torch.no_grad():
                logits = self.model(batch)
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        return predictions

    def save_predictions(self, output_file="predictions.csv"):
        """Generate and save predictions"""
        test_df = pd.read_csv("../datafiles/test_data.csv")
        #test_df = test_df.iloc[:10]  ###
        predictions = self.predict()
        
        results = pd.DataFrame({
            "sequence_name": test_df["sequence_name"],
            "family_id": [self.datamodule.classes[pred] for pred in predictions]
        })
        results.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    # Initialize with legacy patch for version compatibility
    with pl_legacy_patch():
        predictor = Predictor("checkpoints/best_model.ckpt")
        predictor.save_predictions()