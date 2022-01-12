# Automated Cell type classification

Code used in Troullinou, Eirini, et al. "Artificial neural networks in action for an automated cell-type classification of biological neural networks." IEEE Transactions on Emerging Topics in Computational Intelligence (2020).

The user can choose among 1D CNN, RNN, or LSTM models.

Inputs: 
- `data`: matrix `NxD`, where D is the dimension (i.e., time-series) and N the number cells. Ca imaging data, either raw signal or a DF/F transformation.
- `labels`: vector `Nx1`, where N is the number of cells. This vector contains integers from 0 to K, where K denotes the number of classes.

## Train the model
```python
from models import run_cnn_model

output = run_cnn_model(data, labels, epochs=20, num_classes=4, problem_type='multiclass', seed=0)
```

## Make predictions
```python
test_predictions = model.predict_classes(test_data_seq)
```
