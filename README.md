# Autoencoder Anomaly Detection :robot:

This repository contains an example of using a deep learning autoencoder for anomaly detection in data.

## Algorithm Overview :mag_right:

The algorithm performs the following steps:

1. Load the dataset from a CSV file.
2. Scale the features using `StandardScaler` for better training performance.
3. Define and train an autoencoder neural network using PyTorch.
4. Calculate bounds for acceptable values based on average and standard deviation.
5. Detect outliers based on the defined range and label them.
6. Save the labeled data to a new CSV file.

## Requirements :hammer_and_wrench:

- Python (>=3.6)
- PyTorch (>=1.7.0)
- Pandas (>=1.1.0)
- NumPy (>=1.19.0)

## Usage :rocket:

1. Install the required libraries using the following command:
```` python
   pip install torch pandas numpy
````
2. Run the `autoencoder_anomaly_detection.py` script using the jupyter notebook
3. After running the script, you'll find the labeled data in the `outliers_output.csv` file.

## Customization :art:

- You can modify the autoencoder architecture by changing the number of encoding dimensions or adding more layers in the encoder/decoder.

- Adjust the hyperparameters such as learning rate, number of epochs, etc., to optimize the model's performance.

## Results :chart_with_upwards_trend:

The algorithm successfully identifies outliers in the given dataset and labels them based on the calculated range.
