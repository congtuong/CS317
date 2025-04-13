# LSTM for Predictive Maintenance of Turbofan Engines

This repository contains code for developing an LSTM model to predict the Remaining Useful Life (RUL) of turbofan engines using the NASA CMAPSS dataset (specifically FD004). The model is trained to predict when an engine will fail based on sensor data collected during operation.

## Project Overview

The project implements a Long Short-Term Memory (LSTM) neural network to predict the remaining useful life of turbofan engines. FD004 dataset is characterized by engines running under different operating conditions and developing one of two possible faults (HPC Degradation, Fan Degradation).

## Dataset Information

- **Dataset**: FD004
- **Train trajectories**: 248
- **Test trajectories**: 249
- **Operating Conditions**: SIX
- **Fault Modes**: TWO (HPC Degradation, Fan Degradation)

## Requirements

The following packages are required to run the code:

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- TensorFlow
- tqdm
- MLflow
- pyngrok (for MLflow UI tunneling)

All dependencies can be installed using pip. The code is compatible with Python 3.11

To install the required packages, you can create a virtual environment and install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/congtuong/CS317.git
   cd CS317
   ```

2. Install the required packages:

   ```
   pip install pandas numpy matplotlib seaborn scikit-learn tensorflow tqdm mlflow pyngrok
   ```

3. Clone the CMAPSS dataset repository:
   ```
   git clone https://github.com/edwardzjl/CMAPSSData.git
   ```

## Running the Code

1. Open and run the Jupyter notebook:

   ```
   jupyter notebook Lab1_22521624.ipynb
   ```

2. The notebook contains the following main sections:

   - Data loading and preprocessing
   - Baseline model implementation
   - Data visualization and analysis
   - LSTM model implementation
   - Hyperparameter tuning
   - Final model training and evaluation

3. To track experiments with MLflow:

   ```
   mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
   ```

4. Video demonstration of the project is available at: [Drive](https://drive.google.com/file/d/1_cXeHJQ0Vp10Q2IF0lvWfYqOxqxYgmNs/view?usp=sharing)

## Project Structure

- `Lab1_22521624.ipynb`: Main Jupyter notebook containing all the code
- `CMAPSSData/`: Directory containing the dataset files
  - `train_FD004.txt`: Training data
  - `test_FD004.txt`: Test data
  - `RUL_FD004.txt`: Ground truth RUL values for test data
- `simple_lstm.weights.h5`: Saved weights for the simple LSTM model
- `fd004_model.weights.h5`: Saved weights for the final model
- `mlruns/`: Directory containing MLflow experiment tracking data
- `mlruns.db`: SQLite database for MLflow tracking

## Model Architecture

The final LSTM model architecture:

- Masking layer to handle padded sequences
- LSTM layer with 256 units and sigmoid activation
- Dropout layer with 0.1 rate
- Dense output layer

## Hyperparameter Tuning

The notebook includes a random search for hyperparameter tuning with the following parameters:

- Alpha (filter strength)
- Sequence length
- Number of epochs
- Number of layers and nodes per layer
- Dropout rate
- Activation function
- Batch size
- Sensor selection

## Results

The project compares two models for RUL prediction:

1. **Baseline Linear Regression Model**:

   - Serves as a simple benchmark
   - Uses raw sensor data without sequence information
   - RMSE values are calculated on both training and test sets

2. **Tuned LSTM Model**:
   - Final architecture uses a single LSTM layer with 256 units and sigmoid activation
   - Hyperparameter tuning improves performance significantly
   - Optimal parameters: sequence length=30, dropout=0.1, batch size=128, epochs=15
   - Achieves lower RMSE than the base LSTM model (Tuned LSTM model: 26.4 vs. Base LSTM model: 28.9)

The specific RMSE values for both models can be observed by running the notebook. The evaluation metrics include:

- Root Mean Squared Error (RMSE) - lower values indicate better performance
- R-squared (R²) - higher values indicate better performance

## MLflow Tracking

The project uses MLflow to track experiments. To view the experiment results:

1. Start the MLflow UI:

   ```
   mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000
   ```

2. Open your browser and navigate to `http://localhost:5000`

For remote access, the notebook includes code to create an ngrok tunnel to the MLflow UI.

## Acknowledgments

- NASA for providing the CMAPSS dataset
- [edwardzjl](https://github.com/edwardzjl) for the CMAPSS dataset repository
