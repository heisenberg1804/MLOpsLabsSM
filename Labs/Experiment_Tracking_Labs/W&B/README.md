# Lab Submission: Fashion MNIST Model Training Comparison

This document summarizes two experiments conducted to train a Convolutional Neural Network (CNN) on the Fashion MNIST dataset using Weights & Biases (W&B) for experiment tracking. The goal was to compare the performance of an initial model configuration (`neural_network_plus`) against a modified version (`neural_network_modified`) with altered hyperparameters and an enhanced architecture.

## Experiment 1: `neural_network_plus` (Initial Configuration)

### Hyperparameters:
- `dropout`: 0.2
- `layer_1_size`: 32
- `learn_rate`: 0.01
- `momentum`: 0.9
- `epochs`: 5
- `batch_size`: 64
- `sample`: 10000

### Model Architecture:
- Input Layer: `(28, 28, 1)`
- `Conv2D` layer with 32 filters, (5,5) kernel, 'relu' activation
- `MaxPooling2D` layer with (2,2) pool size
- `Dropout` layer with rate 0.2
- `Flatten` layer
- `Dense` output layer with 10 units (for 10 classes), 'softmax' activation

### Observed Performance:
- **Final Validation Accuracy:** 0.8474
- **Final Validation Loss:** 0.4404

## Experiment 2: `neural_network_modified` (Modified Configuration)

### Hyperparameters:
- `dropout`: 0.3 (changed from 0.2)
- `layer_1_size`: 64 (changed from 32)
- `learn_rate`: 0.005 (changed from 0.01)
- `momentum`: 0.9
- `epochs`: 5
- `batch_size`: 64
- `sample`: 10000

### Model Architecture:
- Input Layer: `(28, 28, 1)`
- `Conv2D` layer with 64 filters, (5,5) kernel, 'relu' activation (changed filter count)
- `MaxPooling2D` layer with (2,2) pool size
- **Added `Conv2D` layer with 64 filters, (3,3) kernel, 'relu' activation**
- **Added `MaxPooling2D` layer with (2,2) pool size**
- `Dropout` layer with rate 0.3 (changed dropout rate)
- `Flatten` layer
- `Dense` output layer with 10 units, 'softmax' activation

### Observed Performance:
- **Final Validation Accuracy:** 0.8185
- **Final Validation Loss:** 0.5041

## Comparison and Analysis:

| Metric                   | `neural_network_plus` | `neural_network_modified` |
| :----------------------- | :-------------------- | :------------------------ |
| **Layer 1 Filters**      | 32                    | 64                        |
| **Second Conv Layer**    | No                    | Yes (64 filters)          |
| **Dropout Rate**         | 0.2                   | 0.3                       |
| **Learning Rate**        | 0.01                  | 0.005                     |
| **Final Val Accuracy**   | 0.8474                | 0.8185                    |
| **Final Val Loss**       | 0.4404                | 0.5041                    |

From the comparison, the initial `neural_network_plus` configuration performed slightly better in terms of both final validation accuracy and loss than the `neural_network_modified` version. The modifications in `neural_network_modified`, which included increasing the `layer_1_size`, adding a second convolutional block, and adjusting `dropout` and `learn_rate`, did not lead to an improved performance in this specific set of experiments. This suggests that the initial simpler architecture with its original hyperparameters was more effective for this dataset and training regime, or that the new combination of hyperparameters and architecture might require further tuning (e.g., more epochs, different learning rate schedules, or regularization adjustments) to unlock potential benefits from the increased complexity.