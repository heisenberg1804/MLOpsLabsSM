# TensorFlow Profiler: Model Performance Experiments

This document summarizes the results of various Keras model training experiments on the MNIST dataset, using TensorFlow Profiler via TensorBoard for performance monitoring and comparison.

## Experiments Overview

We conducted three experiments to evaluate the impact of model architecture and hyperparameters on performance.

### Experiment 1: Baseline Model

- **Model Architecture:** A simple Sequential model with one dense layer.
  - Flatten layer: Input shape (28, 28, 1)
  - Dense layer 1: 128 units, 'relu' activation
  - Output Dense layer: 10 units, 'softmax' activation

- **Hyperparameters:**
  - Learning Rate: 0.001
  - Epochs: 5
  - Batch Size: 128

- **Key Performance Metrics:**
  - Training Accuracy: 0.9791
  - Validation Accuracy: 0.9725
  - Training Loss: 0.0726
  - Validation Loss: 0.0926

### Experiment 2: Modified Architecture

- **Model Architecture:** Added an additional dense layer and increased units in the first layer.
  - Flatten layer: Input shape (28, 28, 1)
  - Dense layer 1: 256 units, 'relu' activation
  - Dense layer 2: 64 units, 'relu' activation
  - Output Dense layer: 10 units, 'softmax' activation

- **Hyperparameters:** (Same as Baseline)
  - Learning Rate: 0.001
  - Epochs: 5
  - Batch Size: 128

- **Key Performance Metrics:**
  - Training Accuracy: 0.9897
  - Validation Accuracy: 0.9752
  - Training Loss: 0.0365
  - Validation Loss: 0.0786

### Experiment 3: Hyperparameter Tuning

- **Model Architecture:** (Same as Baseline)
  - Flatten layer: Input shape (28, 28, 1)
  - Dense layer 1: 128 units, 'relu' activation
  - Output Dense layer: 10 units, 'softmax' activation

- **Hyperparameters:** (Modified)
  - Learning Rate: 0.0005
  - Epochs: 10
  - Batch Size: 64

- **Key Performance Metrics:**
  - Training Accuracy: 0.9900
  - Validation Accuracy: 0.9749
  - Training Loss: 0.0389
  - Validation Loss: 0.0831

## Comparative Analysis

- **Experiment 1 (Baseline):** Achieved a validation accuracy of 0.9725 with a relatively simple model. This serves as a good starting point for comparison.

- **Experiment 2 (Modified Architecture):** With an additional layer and more units, the model achieved a validation accuracy of 0.9752. This suggests that a slightly more complex model can improve performance, but it also increases computational cost.

- **Experiment 3 (Hyperparameter Tuning):** By adjusting the learning rate, epochs, and batch size for the baseline architecture, this experiment resulted in a validation accuracy of 0.9749. It demonstrates the importance of hyperparameter tuning, as even with the simpler architecture, optimized hyperparameters can yield competitive results.

Overall, the best performing model in terms of validation accuracy was **Experiment 3** with its tuned hyperparameters, showcasing that careful hyperparameter selection can be as impactful as architectural changes, sometimes even more so for the given task and model complexity. Experiment 2 also showed improvement over the baseline, suggesting that increasing model capacity had a positive effect.

## Conclusion

These experiments highlight the interplay between model architecture and hyperparameters in achieving optimal performance. While increasing model complexity can be beneficial, fine-tuning hyperparameters can also lead to significant improvements, often with less computational overhead in terms of model size. TensorBoard proved invaluable for monitoring these experiments and comparing their outcomes efficiently.
