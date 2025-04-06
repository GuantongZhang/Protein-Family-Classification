# Protein-Family-Classification
A model on protein sequences to perform a multiclass classification task.

In this project, I trained a model on protein sequences to perform a multiclass classification task using the PFam seed dataset. There are several successful models on this type of tasks, such as the pretrained Protein BERT (used as the baseline model in this project).

To improve the accuracy, I used an ESM-2 transformer protein language model. This model outperforms all tested single-sequence protein language models across a range of structure prediction tasks, and enables atomic resolution structure prediction (see https://huggingface.co/docs/transformers/en/model_doc/esm).


## Results

|   epoch |   train_loss |   train_acc |   val_loss |   val_acc |   val_f1 |
|--------:|-------------:|------------:|-----------:|----------:|---------:|
|  1.0000 |       0.0056 |      1.0000 |     3.2160 |    0.0000 |   0.0000 |
|  2.0000 |       0.0008 |      1.0000 |     0.0047 |    1.0000 |   1.0000 |
|  3.0000 |       0.0006 |      1.0000 |     0.0295 |    0.9947 |   0.9947 |
|  4.0000 |       0.0015 |      1.0000 |     0.0035 |    0.9995 |   0.9995 |
|  5.0000 |       0.0005 |      1.0000 |     0.0093 |    0.9986 |   0.9986 |
|  6.0000 |       0.0002 |      1.0000 |     0.0013 |    0.9998 |   0.9998 |
|  7.0000 |       0.0001 |      1.0000 |     0.0045 |    0.9993 |   0.9993 |
|  8.0000 |     nan      |    nan      |     0.0029 |    0.9993 |   0.9993 |

During the training process, the early stop trigger has been set with patience=2, evaluated by the validation loss value, and it stopped at the 8th epoch. The highest validation accuracy, 0.9998, was found at the 6th epoch, together with the least validation loss, 0.0013. See /new_code/extracted_training_curves.png for the visualization of the training process. Note that the train_loss and train_acc were not recorded due to the early stop, and it doesn’t affect the final conclusion.
