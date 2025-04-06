import numpy as np
import pandas as pd
import torch
import pandas as pd
import matplotlib.pyplot as plt

checkpoint = torch.load("checkpoints/protein_classifier.pt")
training_history = checkpoint['training_history']

max_length = max(len(v) for v in training_history.values())
for key in training_history:
    if len(training_history[key]) < max_length:
        training_history[key] = np.pad(
            training_history[key],
            (0, max_length - len(training_history[key])),
            mode='constant',
            constant_values=np.nan
        )

# print table
df_history = pd.DataFrame(training_history)
df_history.insert(0, 'epoch', range(1, len(df_history)+1))
print(df_history.to_markdown(index=False, floatfmt=".4f"))

# plot
start_idx = 1  # omit the first epoch for better plot
truncated_history = {k: v[start_idx:] for k, v in training_history.items()}  
df_history = pd.DataFrame(truncated_history)
df_history.insert(0, 'epoch', range(1+start_idx, len(df_history)+1+start_idx))

plt.figure(figsize=(12, 4))

# plot loss curve
plt.subplot(1, 2, 1)
plt.plot(df_history['epoch'], df_history['train_loss'], label='Train')
plt.plot(df_history['epoch'], df_history['val_loss'], label='Validation')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# plot accuracy curve
plt.subplot(1, 2, 2)
plt.plot(df_history['epoch'], df_history['train_acc'], label='Train')
plt.plot(df_history['epoch'], df_history['val_acc'], label='Validation')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('extracted_training_curves.png')  # save figure
plt.show()