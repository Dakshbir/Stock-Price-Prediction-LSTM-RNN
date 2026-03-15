"""
Enhanced training module for Tesla stock price prediction using RNN and LSTM models.
This module provides comprehensive training functions with improved architecture and evaluation.
"""

# Import necessary libraries and modules
from datetime import datetime
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader.data import DataReader
from pandas_datareader import data as pdr
import pandas_ta as ta
import os

# Import Keras layers and models
from keras.layers import LSTM, SimpleRNN, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Import custom utility functions
from .utils import split_sequence, calculate_metrics, plot_predictions

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Global parameters
N_STEPS = 60  # Increased from 1 to 60 for better temporal pattern capture
FEATURES = 1
MV_N_STEPS = 30  # For multivariate models

def sequence_generation(dataset: pd.DataFrame, sc: MinMaxScaler, model: Sequential, 
                       steps_future: int, test_set, n_steps: int = N_STEPS):
    """
    Generate future stock price predictions using sequence generation approach.
    
    Args:
        dataset (pd.DataFrame): Original dataset containing stock data
        sc (MinMaxScaler): Fitted MinMaxScaler for data transformation
        model (Sequential): Trained Keras model
        steps_future (int): Number of future steps to predict
        test_set (np.array): Test dataset
        n_steps (int): Number of time steps for input sequence
    
    Returns:
        np.array: Generated sequence of future predictions
    """
    try:
        # Extract the 'High' column for the last portion of data
        high_dataset = dataset.iloc[len(dataset) - len(test_set) - n_steps:]["High"]
        
        # Scale the 'High' column using the MinMaxScaler
        high_dataset = sc.transform(high_dataset.values.reshape(-1, 1))
        
        # Initialize the inputs with the last n_steps values
        inputs = high_dataset[-n_steps:]
        
        # Generate predictions for steps_future time steps into the future
        predictions = []
        current_input = inputs.copy()
        
        for i in range(steps_future):
            # Predict the next value based on the previous n_steps values
            curr_pred = model.predict(current_input.reshape(1, n_steps, FEATURES), verbose=0)
            predictions.append(curr_pred[0, 0])
            
            # Update the input sequence by removing the first element and adding the prediction
            current_input = np.append(current_input[1:], curr_pred, axis=0)
        
        # Convert predictions to numpy array and inverse transform
        predictions = np.array(predictions).reshape(-1, 1)
        return sc.inverse_transform(predictions)
        
    except Exception as e:
        print(f"Error in sequence generation: {str(e)}")
        return None

def train_rnn_model(X_train, y_train, n_steps, features, sc, test_set, dataset, 
                   epochs=50, batch_size=32, verbose=1, steps_in_future=30, 
                   save_model_path=None, validation_split=0.2):
    """
    Train an enhanced RNN model for Tesla stock price prediction.
    
    Args:
        X_train (np.array): Training input sequences
        y_train (np.array): Training target values
        n_steps (int): Number of time steps in input sequences
        features (int): Number of features per time step
        sc (MinMaxScaler): Fitted MinMaxScaler
        test_set (np.array): Test dataset
        dataset (pd.DataFrame): Original dataset
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        verbose (int): Verbosity level
        steps_in_future (int): Number of future steps to predict
        save_model_path (str): Path to save the trained model
        validation_split (float): Fraction of data to use for validation
    
    Returns:
        Sequential: Trained RNN model
    """
    print("="*60)
    print("TRAINING ENHANCED RNN MODEL FOR TESLA STOCK PREDICTION")
    print("="*60)
    
    try:
        # Create enhanced RNN model architecture
        model = Sequential([
            # First RNN layer with return_sequences=True to stack layers
            SimpleRNN(units=50, return_sequences=True, input_shape=(n_steps, features)),
            Dropout(0.2),
            
            # Second RNN layer
            SimpleRNN(units=50, return_sequences=True),
            Dropout(0.2),
            
            # Third RNN layer
            SimpleRNN(units=50),
            Dropout(0.2),
            
            # Dense layers for final prediction
            Dense(units=25, activation='relu'),
            Dense(units=1)
        ])
        
        # Compile model with improved optimizer
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=['mae']
        )
        
        # Print model summary
        print("RNN Model Architecture:")
        model.summary()
        
        # Define callbacks for better training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        # Train the model
        print(f"Training RNN model for {epochs} epochs...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Prepare test data
        inputs = sc.transform(test_set.reshape(-1, 1))
        X_test, y_test = split_sequence(inputs, n_steps)
        X_test = X_test.reshape(-1, n_steps, features)
        
        # Make predictions
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        y_test_actual = sc.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate and display metrics
        print("\nRNN Model Performance:")
        metrics = calculate_metrics(y_test_actual, predicted_stock_price)
        
        # Generate future predictions
        print(f"\nGenerating {steps_in_future} future predictions...")
        future_results = sequence_generation(dataset, sc, model, steps_in_future, test_set, n_steps)
        
        if future_results is not None:
            print("Future predictions generated successfully!")
            
            # Plot predictions if possible
            if len(y_test_actual) >= steps_in_future:
                plot_predictions(
                    y_test_actual[:steps_in_future], 
                    future_results, 
                    "Tesla Stock Price - RNN Future Predictions",
                    save_path="output/figures/rnn_future_predictions.png" if save_model_path else None
                )
        
        # Save model if path is provided
        if save_model_path:
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            model.save(save_model_path)
            print(f"RNN model saved to: {save_model_path}")
        
        return model
        
    except Exception as e:
        print(f"Error in RNN training: {str(e)}")
        return None

def train_lstm_model(X_train, y_train, n_steps, features, sc, test_set, dataset,
                    epochs=50, batch_size=32, verbose=1, steps_in_future=30,
                    save_model_path=None, validation_split=0.2):
    """
    Train an enhanced LSTM model for Tesla stock price prediction.
    
    Args:
        X_train (np.array): Training input sequences
        y_train (np.array): Training target values
        n_steps (int): Number of time steps in input sequences
        features (int): Number of features per time step
        sc (MinMaxScaler): Fitted MinMaxScaler
        test_set (np.array): Test dataset
        dataset (pd.DataFrame): Original dataset
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        verbose (int): Verbosity level
        steps_in_future (int): Number of future steps to predict
        save_model_path (str): Path to save the trained model
        validation_split (float): Fraction of data to use for validation
    
    Returns:
        Sequential: Trained LSTM model
    """
    print("="*60)
    print("TRAINING ENHANCED LSTM MODEL FOR TESLA STOCK PREDICTION")
    print("="*60)
    
    try:
        # Create enhanced LSTM model architecture
        model = Sequential([
            # First LSTM layer with return_sequences=True to stack layers
            LSTM(units=50, return_sequences=True, input_shape=(n_steps, features)),
            Dropout(0.2),
            
            # Second LSTM layer
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            
            # Third LSTM layer
            LSTM(units=50),
            Dropout(0.2),
            
            # Dense layers for final prediction
            Dense(units=25, activation='relu'),
            Dense(units=1)
        ])
        
        # Compile model with improved optimizer
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=['mae']
        )
        
        # Print model summary
        print("LSTM Model Architecture:")
        model.summary()
        
        # Define callbacks for better training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        # Train the model
        print(f"Training LSTM model for {epochs} epochs...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Prepare test data
        inputs = sc.transform(test_set.reshape(-1, 1))
        X_test, y_test = split_sequence(inputs, n_steps)
        X_test = X_test.reshape(-1, n_steps, features)
        
        # Make predictions
        predicted_stock_price = model.predict(X_test)
        predicted_stock_price = sc.inverse_transform(predicted_stock_price)
        y_test_actual = sc.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate and display metrics
        print("\nLSTM Model Performance:")
        metrics = calculate_metrics(y_test_actual, predicted_stock_price)
        
        # Generate future predictions
        print(f"\nGenerating {steps_in_future} future predictions...")
        future_results = sequence_generation(dataset, sc, model, steps_in_future, test_set, n_steps)
        
        if future_results is not None:
            print("Future predictions generated successfully!")
            
            # Plot predictions if possible
            if len(y_test_actual) >= steps_in_future:
                plot_predictions(
                    y_test_actual[:steps_in_future], 
                    future_results, 
                    "Tesla Stock Price - LSTM Future Predictions",
                    save_path="output/figures/lstm_future_predictions.png" if save_model_path else None
                )
        
        # Save model if path is provided
        if save_model_path:
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            model.save(save_model_path)
            print(f"LSTM model saved to: {save_model_path}")
        
        return model
        
    except Exception as e:
        print(f"Error in LSTM training: {str(e)}")
        return None

def create_multivariate_sequences(X, y, n_steps):
    """
    Create sequences for multivariate time series data.
    
    Args:
        X (np.array): Feature data
        y (np.array): Target data
        n_steps (int): Number of time steps
    
    Returns:
        tuple: (X_seq, y_seq) - Sequenced feature and target data
    """
    X_seq, y_seq = [], []
    for i in range(n_steps, len(X)):
        X_seq.append(X[i-n_steps:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def train_multivariate_lstm(X_train, y_train, X_test, y_test, mv_features, mv_sc,
                           epochs=50, batch_size=32, verbose=1, save_model_path=None,
                           validation_split=0.2):
    """
    Train an enhanced multivariate LSTM model for Tesla stock price prediction.
    
    Args:
        X_train (np.array): Training input sequences
        y_train (np.array): Training target values
        X_test (np.array): Test input sequences
        y_test (np.array): Test target values
        mv_features (int): Number of multivariate features
        mv_sc (MinMaxScaler): Fitted MinMaxScaler for multivariate data
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        verbose (int): Verbosity level
        save_model_path (str): Path to save the trained model
        validation_split (float): Fraction of data to use for validation
    
    Returns:
        Sequential: Trained multivariate LSTM model
    """
    print("="*60)
    print("TRAINING ENHANCED MULTIVARIATE LSTM MODEL FOR TESLA")
    print("="*60)
    
    try:
        # Create enhanced multivariate LSTM model
        model_mv = Sequential([
            # First LSTM layer
            LSTM(units=100, return_sequences=True, input_shape=(MV_N_STEPS, mv_features)),
            Dropout(0.3),
            
            # Second LSTM layer
            LSTM(units=100, return_sequences=True),
            Dropout(0.3),
            
            # Third LSTM layer
            LSTM(units=50),
            Dropout(0.3),
            
            # Dense layers
            Dense(units=50, activation='relu'),
            Dropout(0.2),
            Dense(units=25, activation='relu'),
            Dense(units=1)
        ])
        
        # Compile the model
        model_mv.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=['mae']
        )
        
        # Print model summary
        print("Multivariate LSTM Model Architecture:")
        model_mv.summary()
        
        # Define callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        ]
        
        # Train the model
        print(f"Training Multivariate LSTM model for {epochs} epochs...")
        history = model_mv.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=verbose
        )
        
        # Make predictions
        predictions = model_mv.predict(X_test)
        
        # Calculate and display metrics
        print("\nMultivariate LSTM Model Performance:")
        metrics = calculate_metrics(y_test.reshape(-1, 1), predictions)
        
        # Plot predictions
        plot_predictions(
            y_test, 
            predictions.flatten(), 
            "Tesla Stock Price - Multivariate LSTM",
            save_path="output/figures/mv_lstm_predictions.png" if save_model_path else None
        )
        
        # Save model if path is provided
        if save_model_path:
            os.makedirs(os.path.dirname(save_model_path), exist_ok=True)
            model_mv.save(save_model_path)
            print(f"Multivariate LSTM model saved to: {save_model_path}")
        
        return model_mv
        
    except Exception as e:
        print(f"Error in Multivariate LSTM training: {str(e)}")
        return None

def compare_models(models_dict, test_data_dict):
    """
    Compare performance of different models and generate comparison report.
    
    Args:
        models_dict (dict): Dictionary of trained models
        test_data_dict (dict): Dictionary of test data for each model
    
    Returns:
        pd.DataFrame: Comparison results
    """
    print("="*60)
    print("TESLA STOCK PREDICTION - MODEL COMPARISON")
    print("="*60)
    
    comparison_results = []
    
    for model_name, model in models_dict.items():
        if model is not None and model_name in test_data_dict:
            try:
                test_data = test_data_dict[model_name]
                # This would need to be implemented based on specific test data format
                # For now, we'll create a placeholder
                comparison_results.append({
                    'Model': model_name,
                    'Parameters': model.count_params(),
                    'Architecture': f"{len(model.layers)} layers"
                })
            except Exception as e:
                print(f"Error comparing {model_name}: {str(e)}")
    
    if comparison_results:
        comparison_df = pd.DataFrame(comparison_results)
        print(comparison_df.to_string(index=False))
        
        # Save comparison results
        os.makedirs("output", exist_ok=True)
        comparison_df.to_csv("output/tesla_model_comparison.csv", index=False)
        print("Model comparison saved to output/tesla_model_comparison.csv")
        
        return comparison_df
    else:
        print("No valid models to compare.")
        return None

# Additional utility functions for model training
def plot_training_history(history, title="Training History", save_path=None):
    """
    Plot training history including loss and metrics.
    
    Args:
        history: Keras training history object
        title (str): Title for the plot
        save_path (str): Path to save the plot
    """
    plt.figure(figsize=(15, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot training and validation MAE if available
    if 'mae' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        if 'val_mae' in history.history:
            plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title(f'{title} - MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()

def save_model_summary(model, model_name, save_path):
    """
    Save model architecture summary to a text file.
    
    Args:
        model: Keras model
        model_name (str): Name of the model
        save_path (str): Path to save the summary
    """
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(f"Model: {model_name}\n")
            f.write("="*50 + "\n")
            model.summary(print_fn=lambda x: f.write(x + '\n'))
            f.write(f"\nTotal Parameters: {model.count_params():,}\n")
        print(f"Model summary saved to: {save_path}")
    except Exception as e:
        print(f"Error saving model summary: {str(e)}")
