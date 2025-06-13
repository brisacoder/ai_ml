# Understanding `car_telemetry_transformers.ipynb`

This document walks through every step of the notebook in detail. The goal is to explain how synthetic car telemetry data can be processed with a Transformer encoder for simple anomaly detection.

The notebook is organised into nine sections. Below each section is explained line by line.

## 1. Import Libraries

```python
import pandas as pd
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
```

* **pandas** and **numpy** provide data manipulation utilities.
* **torch** and **torch.nn** bring PyTorch's tensor operations and neural network layers.
* **matplotlib.pyplot** is used for visualising the synthetic data and the model output.

## 2. Simulate Telemetry Data

```python
np.random.seed(42)
time_steps = 100

data = {
    "timestamp": pd.date_range("2025-01-01", periods=time_steps, freq="1s"),
    "rpm": np.random.normal(3000, 300, time_steps),
    "speed": np.random.normal(60, 5, time_steps),
    "gear": np.random.randint(1, 6, time_steps),
    "brake": np.random.choice([0, 1], time_steps, p=[0.9, 0.1]),
    "coolant_temp": np.random.normal(190, 5, time_steps),
    "throttle_pos": np.random.uniform(0, 100, time_steps)
}

df = pd.DataFrame(data)
```

* **`np.random.seed(42)`** ensures that the generated random numbers are the same each time you run the notebook, which helps with reproducibility.
* **`time_steps`** defines the length of the time series. Each step is one second apart.
* A dictionary named `data` is created:
  * `timestamp` uses `pd.date_range` to build 100 consecutive timestamps starting from 2025-01-01 with one second between entries.
  * `rpm` is drawn from a normal distribution with mean 3000 revolutions per minute and standard deviation 300.
  * `speed` is also from a normal distribution, centred at 60 mph with a deviation of 5.
  * `gear` is a random integer from 1 to 5 (the upper bound 6 is exclusive).
  * `brake` is a binary flag (0 or 1) with a 10% chance of being 1.
  * `coolant_temp` simulates engine coolant temperature around 190°F.
* `pd.DataFrame(data)` converts the dictionary to a table, giving us a 100-row dataset of synthetic telemetry.

## 3. Normalize Features

```python
features = ["rpm", "speed", "gear", "brake", "coolant_temp", "throttle_pos"]
normalized = (df[features] - df[features].mean()) / df[features].std()
```

* The list `features` selects the numeric columns we want to feed the model.
* Normalisation (subtract the mean then divide by the standard deviation) gives every feature zero mean and unit variance. This is common when training neural networks so that each input dimension contributes roughly equally.

## 4. Positional Encoding

```python
T, D = normalized.shape

position_ids = torch.arange(T).unsqueeze(1)  # (T, 1)
position_embedding_layer = nn.Embedding(T, D)
position_embeddings = position_embedding_layer(position_ids.squeeze())
```

* `T` is the number of time steps (100) and `D` is the number of features (6).
* `torch.arange(T)` creates a tensor `[0, 1, 2, ..., 99]`. `unsqueeze(1)` changes its shape from `(T,)` to `(T, 1)` so each index stands alone.
* `nn.Embedding(T, D)` creates a learnable table of `T` rows and `D` columns. Each row provides a D‑dimensional vector that represents a specific time step.
* Calling `position_embedding_layer(position_ids.squeeze())` looks up each index and returns a tensor of shape `(T, D)` containing positional vectors. These embeddings let the model know the order of the sequence since a Transformer has no inherent sense of time.

## 5. Convert to Tensor and Add Positional Embeddings

```python
inputs = torch.tensor(normalized.values, dtype=torch.float32)  # (T, D)
transformer_input = inputs + position_embeddings  # (T, D)
```

* `normalized.values` gives us the underlying NumPy array of the DataFrame. Converting it to a float tensor yields shape `(100, 6)`.
* Adding the positional embeddings elementwise combines feature information with timing information. The resulting tensor `transformer_input` is what we feed into the model.

## 6. Visualize a Sample

```python
print("Transformer input shape:", transformer_input.shape)
print("First few rows:\n", transformer_input[:5])

plt.plot(df["timestamp"], df["rpm"], label="RPM")
plt.plot(df["timestamp"], df["speed"], label="Speed")
plt.legend()
plt.title("Synthetic Telemetry")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

* The shape printout confirms the dimensions of `(100, 6)`.
* The first five rows offer a glimpse at the combined numeric and positional information.
* The plot of RPM and speed over time provides intuition for the simulated data—useful for verifying that the signals look plausible.

## 7. Transformer Encoder Model

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleTransformerEncoder(nn.Module):
    def __init__(self, input_dim, nhead, num_layers, dim_feedforward):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = x.unsqueeze(1)  # (seq_len, 1, input_dim)
        x = self.transformer(x)
        x = x.squeeze(1)    # back to (seq_len, input_dim)
        return self.output_layer(x)

model = SimpleTransformerEncoder(
    input_dim=transformer_input.shape[1], nhead=2, num_layers=2, dim_feedforward=128
)
```

* The class defines a very small Transformer encoder.
* `nn.TransformerEncoderLayer` is PyTorch’s built-in implementation of a single Transformer block. We specify:
  * `d_model=input_dim` — the size of each input vector (5 in this notebook).
  * `nhead` — how many attention heads to use; here it is 2.
  * `dim_feedforward` — the hidden size in the layer’s feed‑forward network.
* `nn.TransformerEncoder` then stacks multiple such layers (`num_layers=2`).
* `self.output_layer` is a linear layer mapping the encoder output back to the same dimensionality so we can compare it with the input.
* In `forward`, the tensor is temporarily expanded to shape `(sequence_length, batch_size=1, input_dim)` because PyTorch’s Transformer expects the batch dimension. After the Transformer processes the sequence we squeeze the dimension away again.

## 8. Forward Pass and Anomaly Score Calculation

```python
model.eval()
with torch.no_grad():
    reconstructed = model(transformer_input)

loss_per_timestep = F.mse_loss(reconstructed, transformer_input, reduction='none').mean(dim=1)
anomaly_scores = loss_per_timestep.detach().numpy()

df["anomaly_score"] = anomaly_scores
```

* `model.eval()` sets layers like dropout or batch norm (not used here, but good practice) to evaluation mode.
* `torch.no_grad()` ensures gradients are not tracked, reducing memory usage during inference.
* The model processes the sequence and outputs another `(100, 6)` tensor named `reconstructed`.
* The mean squared error between the reconstructed tensor and the input is computed for each time step. High values mean the model struggled to recreate that point and could indicate an anomaly.
* The resulting score is stored back into the DataFrame under a new column `anomaly_score` for later inspection or plotting.

## 9. Visualize Results

```python
plt.figure(figsize=(10, 4))
plt.plot(df["timestamp"], df["anomaly_score"], label="Anomaly Score")
plt.title("Telemetry Transformer Anomaly Detection")
plt.xlabel("Time")
plt.ylabel("Score")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend()
plt.show()
```

* This final plot shows how the anomaly score changes over time. In this synthetic example there may not be pronounced anomalies, but peaks in the curve would correspond to out‑of‑pattern telemetry readings.

---

This walkthrough is intended to provide step‑by‑step insight into the PyTorch operations used in `car_telemetry_transformers.ipynb`. By synthesising data, normalising it, embedding positional information, and passing it through a Transformer encoder, we obtain a simple anomaly score for each time step. The code can serve as a starting point for experimenting with Transformer models on tabular or time‑series data in PyTorch.
