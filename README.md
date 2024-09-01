# Multi-Task Learning (MTL) Model for Sine and Cosine Prediction

This repository contains a PyTorch implementation of a Multi-Task Learning (MTL) model designed to predict both sine and cosine values from a single input. The model uses shared layers to learn general features from the input and task-specific branches to make precise predictions for each task.

## Table of Contents

- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
  - [Shared Layers](#shared-layers)
  - [Task-Specific Layers](#task-specific-layers)
  - [Forward Pass](#forward-pass)

## Introduction

This project demonstrates the power of Multi-Task Learning (MTL) by building a neural network that can simultaneously predict sine and cosine values from the same input. By leveraging shared layers, the model efficiently learns representations that are beneficial for both tasks, improving overall performance and computational efficiency.

## Model Architecture

### Shared Layers

The shared layers in the model extract general features from the input data. These layers ensure that the model learns a representation of the input that is useful for both tasks (predicting sine and cosine values).

```python
self.model = torch.nn.Sequential(
    torch.nn.Linear(1, h),
    torch.nn.ReLU(),
    torch.nn.Linear(h, h),
    torch.nn.ReLU()
)
```

### Task-Specific Layers

The task-specific layers specialize in generating predictions for sine and cosine, respectively. Each branch takes the shared representation and fine-tunes it for the particular task.

```python
# Sine branch
self.model_sin = torch.nn.Sequential(
    torch.nn.Linear(h, h),
    torch.nn.ReLU(),
    torch.nn.Linear(h, 1)
)

# Cosine branch
self.model_cos = torch.nn.Sequential(
    torch.nn.Linear(h, h),
    torch.nn.ReLU(),
    torch.nn.Linear(h, 1)
)
```

### Forward Pass

During the forward pass, the input is processed through the shared layers to generate a common feature representation. This representation is then passed to each task-specific branch to produce both sine and cosine predictions.

```python
def forward(self, inputs):
    # pass through shared layers
    x1 = self.model(inputs)

    # generate sin(x) prediction
    output_sin = self.model_sin(x1)

    # generate cos(x) prediction
    output_cos = self.model_cos(x1)

    # return both predictions
    return output_sin, output_cos
```
