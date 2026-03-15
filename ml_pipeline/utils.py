"""
Enhanced utility functions for Tesla stock price prediction.
This module provides comprehensive data processing, evaluation, and visualization utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_ta as ta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def train_test_split(dataset, tstart, tend, columns=['High']):
    """
    Split dataset into training and test sets based on time periods.
    
    Args:
        dataset (pd.DataFrame): Stock price dataset
        tstart (int): Start year for training data
        tend (int): End year for training data
        columns (list): List of columns to include in the split
    
    Returns:
        tuple: (train, test) - Training and test data arrays
    """
    try:
        # Select data within the specified time range for training
        train = dataset.loc[f"{tstart}":f"{tend}", columns].values
        
        # Select data after the specified end year for testing
        test = dataset.loc[f"{tend+1}":, columns].values
        
        print(f"Training data shape: {train.shape}")
        print(f"Test data shape: {test.shape}")
        print(f"Training period: {tstart} to {tend}")
        print(f"Test period: {tend+1} onwards")
        
        return train, test
        
    except Exception as e:
        print(f"Error in train_test_split: {str(e)}")
        return None, None

def split_sequence(sequence, n_steps):
    """
    Split a sequence into input-output pairs for time series forecasting.
    
    Args:
        sequence (np.array): Input sequence data
        n_steps (int): Number of time steps to look back
    
    Returns:
        tuple: (X, y) - Input sequences and corresponding targets
    """
    X, y = list(), list()
    
    for i in range(len(sequence)):
        # Find the end of this pattern
        end_ix = i + n_steps
        
        # Check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
            
        # Gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    
    return np.array(X), np.array(y)

def calculate_metrics(test, predicted):
    """
    Calculate comprehensive evaluation metrics for stock price predictions.
    
    Args:
        test (np.array): Actual test values
        predicted (np.array): Predicted values
    
    Returns:
        dict: Dictionary containing calculated metrics
    """
    try:
        # Ensure arrays are flattened and same shape
        test = test.flatten()
        predicted = predicted.flatten()
        
        # Calculate RMSE (Root Mean Squared Error)
        rmse = np.sqrt(mean_squared_error(test, predicted))
        
        # Calculate MAE (Mean Absolute Error)
        mae = mean_absolute_error(test, predicted)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        non_zero_mask = test != 0
        if np.sum(non_zero_mask) > 0:
            mape = np.mean(np.abs((test[non_zero_mask] - predicted[non_zero_mask]) / test[non_zero_mask])) * 100
        else:
            mape = float('inf')
        
        # Calculate directional accuracy
        if len(test) > 1 and len(predicted) > 1:
            test_direction = np.diff(test) > 0
            pred_direction = np.diff(predicted) > 0
            directional_accuracy = np.mean(test_direction == pred_direction) * 100
        else:
            directional_accuracy = 0
        
        # Calculate R-squared
        ss_res = np.sum((test - predicted) ** 2)
        ss_tot = np.sum((test - np.mean(test)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        print("="*50)
        print("MODEL EVALUATION METRICS")
        print("="*50)
        print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Directional Accuracy: {directional_accuracy:.2f}%")
        print(f"R-squared Score: {r_squared:.4f}")
        print("="*50)
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy,
            'R_squared': r_squared
        }
        
    except Exception as e:
        print(f"Error calculating metrics: {str(e)}")
        return None

def plot_predictions(test, predicted, title, save_path=None):
    """
    Plot real and predicted values for time series forecasting.
    
    Args:
        test (np.array): Actual test values
        predicted (np.array): Predicted values
        title (str): Title for the plot
        save_path (str): Path to save the plot (optional)
    """
    try:
        plt.figure(figsize=(16, 8))
        
        # Ensure arrays are flattened
        test = test.flatten()
        predicted = predicted.flatten()
        
        # Plot the real values in blue
        plt.plot(test, color="blue", label="Actual", linewidth=2, alpha=0.8)
        
        # Plot the predicted values in red
        plt.plot(predicted, color="red", label="Predicted", linewidth=2, alpha=0.8)
        
        # Set the title and labels for the plot
        plt.title(f'{title}', fontsize=16, fontweight='bold')
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Stock Price ($)", fontsize=12)
        
        # Add a legend to differentiate real and predicted values
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add some statistics to the plot
        rmse = np.sqrt(mean_squared_error(test, predicted))
        mae = mean_absolute_error(test, predicted)
        plt.text(0.02, 0.98, f'RMSE: ${rmse:.2f}\nMAE: ${mae:.2f}', 
                transform=plt.gca().transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Save plot if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error plotting predictions: {str(e)}")

def process_and_split_multivariate_data(dataset, tstart, tend, mv_features):
    """
    Process and split multivariate data with enhanced technical indicators for Tesla.
    
    Args:
        dataset (pd.DataFrame): Original stock dataset
        tstart (int): Start year for training
        tend (int): End year for training
        mv_features (int): Number of multivariate features
    
    Returns:
        tuple: (X_train, y_train, X_test, y_test, mv_sc) - Processed multivariate data and scaler
    """
    try:
        print("Processing multivariate data for Tesla...")
        
        # Create a copy of the dataset
        multi_variate_df = dataset.copy()
        
        # Calculate enhanced technical indicators
        print("Calculating technical indicators...")
        
        # RSI with different periods
        multi_variate_df['RSI'] = ta.rsi(multi_variate_df.Close, length=14)
        
        # Multiple EMAs for different time horizons
        multi_variate_df['EMA_12'] = ta.ema(multi_variate_df.Close, length=12)
        multi_variate_df['EMA_26'] = ta.ema(multi_variate_df.Close, length=26)
        multi_variate_df['EMA_50'] = ta.ema(multi_variate_df.Close, length=50)
        
        # MACD indicator
        macd_data = ta.macd(multi_variate_df.Close)
        if 'MACD_12_26_9' in macd_data.columns:
            multi_variate_df['MACD'] = macd_data['MACD_12_26_9']
        else:
            multi_variate_df['MACD'] = ta.sma(multi_variate_df.Close, length=12) - ta.sma(multi_variate_df.Close, length=26)
        
        # Bollinger Bands
        bb_data = ta.bbands(multi_variate_df.Close, length=20)
        if 'BBU_20_2.0' in bb_data.columns:
            multi_variate_df['BB_Upper'] = bb_data['BBU_20_2.0']
            multi_variate_df['BB_Lower'] = bb_data['BBL_20_2.0']
        else:
            # Fallback calculation
            sma_20 = ta.sma(multi_variate_df.Close, length=20)
            std_20 = multi_variate_df.Close.rolling(window=20).std()
            multi_variate_df['BB_Upper'] = sma_20 + (std_20 * 2)
            multi_variate_df['BB_Lower'] = sma_20 - (std_20 * 2)
        
        # Volume-based indicator
        multi_variate_df['Volume_SMA'] = ta.sma(multi_variate_df.Volume, length=20)
        
        # Create target variable
        multi_variate_df['Target'] = multi_variate_df['Adj Close'] - multi_variate_df['Open']
        multi_variate_df['Target'] = multi_variate_df['Target'].shift(-1)
        
        # Drop rows with missing values
        multi_variate_df.dropna(inplace=True)
        
        # Remove unnecessary columns
        multi_variate_df.drop(['Volume', 'Close'], axis=1, inplace=True)
        
        print(f"Multivariate dataset shape after preprocessing: {multi_variate_df.shape}")
        
        # Define feature columns (adjust based on mv_features)
        if mv_features == 8:
            feat_columns = ['Open', 'High', 'RSI', 'EMA_12', 'EMA_26', 'EMA_50', 'MACD', 'BB_Upper']
        elif mv_features == 6:
            feat_columns = ['Open', 'High', 'RSI', 'EMA_12', 'EMA_26', 'EMA_50']
        else:
            # Default to first mv_features columns
            available_cols = ['Open', 'High', 'RSI', 'EMA_12', 'EMA_26', 'EMA_50', 'MACD', 'BB_Upper']
            feat_columns = available_cols[:mv_features]
        
        label_col = ['Target']
        
        # Split train and test data
        mv_training_set, mv_test_set = train_test_split(
            multi_variate_df, tstart, tend, feat_columns + label_col
        )
        
        if mv_training_set is None or mv_test_set is None:
            return None, None, None, None, None
        
        # Extract features and labels
        X_train = mv_training_set[:, :-1]
        y_train = mv_training_set[:, -1]
        X_test = mv_test_set[:, :-1]
        y_test = mv_test_set[:, -1]
        
        # Scale the data
        mv_sc = MinMaxScaler(feature_range=(0, 1))
        X_train_scaled = mv_sc.fit_transform(X_train)
        X_test_scaled = mv_sc.transform(X_test)
        
        print(f"Multivariate data processed successfully!")
        print(f"Training features shape: {X_train_scaled.shape}")
        print(f"Test features shape: {X_test_scaled.shape}")
        
        return X_train_scaled, y_train, X_test_scaled, y_test, mv_sc
        
    except Exception as e:
        print(f"Error processing multivariate data: {str(e)}")
        return None, None, None, None, None

def create_sequences_multivariate(X, y, n_steps):
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

def plot_technical_indicators(dataset, tstart, tend, save_path=None):
    """
    Plot technical indicators for Tesla stock analysis.
    
    Args:
        dataset (pd.DataFrame): Stock dataset with technical indicators
        tstart (int): Start year for plotting
        tend (int): End year for plotting
        save_path (str): Path to save the plot
    """
    try:
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # Plot 1: Stock Price with EMAs
        dataset.loc[f"{tstart}":f"{tend}", ['High', 'EMA_12', 'EMA_26', 'EMA_50']].plot(
            ax=axes[0], title="Tesla Stock Price with Moving Averages")
        axes[0].set_ylabel("Price ($)")
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: RSI
        dataset.loc[f"{tstart}":f"{tend}", 'RSI'].plot(
            ax=axes[1], title="Tesla RSI", color='orange')
        axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        axes[1].set_ylabel("RSI")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: MACD
        if 'MACD' in dataset.columns:
            dataset.loc[f"{tstart}":f"{tend}", 'MACD'].plot(
                ax=axes[2], title="Tesla MACD", color='purple')
            axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
            axes[2].set_ylabel("MACD")
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Technical indicators plot saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error plotting technical indicators: {str(e)}")

def save_predictions_to_csv(actual, predicted, dates=None, save_path="output/predictions.csv"):
    """
    Save predictions to CSV file for further analysis.
    
    Args:
        actual (np.array): Actual values
        predicted (np.array): Predicted values
        dates (list): List of dates (optional)
        save_path (str): Path to save the CSV file
    """
    try:
        # Create DataFrame
        df = pd.DataFrame({
            'Actual': actual.flatten(),
            'Predicted': predicted.flatten(),
            'Error': actual.flatten() - predicted.flatten(),
            'Absolute_Error': np.abs(actual.flatten() - predicted.flatten())
        })
        
        if dates is not None:
            df['Date'] = dates
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        print(f"Predictions saved to {save_path}")
        
        # Print summary statistics
        print("\nPrediction Summary:")
        print(f"Mean Absolute Error: ${df['Absolute_Error'].mean():.2f}")
        print(f"Max Error: ${df['Error'].max():.2f}")
        print(f"Min Error: ${df['Error'].min():.2f}")
        
    except Exception as e:
        print(f"Error saving predictions: {str(e)}")

def load_tesla_data(start_date='2012-01-01', end_date=None):
    """
    Load Tesla stock data with error handling and data quality checks.
    
    Args:
        start_date (str): Start date for data loading
        end_date (str): End date for data loading (default: current date)
    
    Returns:
        pd.DataFrame: Tesla stock data
    """
    try:
        from pandas_datareader import data as pdr
        import yfinance as yf
        
        # Override Yahoo Finance downloader
        yf.pdr_override()
        
        if end_date is None:
            end_date = datetime.now()
        
        print(f"Loading Tesla (TSLA) stock data from {start_date} to {end_date}...")
        
        # Load data
        dataset = pdr.get_data_yahoo('TSLA', start=start_date, end=end_date)
        
        print(f"Tesla stock data loaded successfully!")
        print(f"Data shape: {dataset.shape}")
        print(f"Date range: {dataset.index[0]} to {dataset.index[-1]}")
        
        # Check for missing values
        missing_values = dataset.isnull().sum()
        if missing_values.sum() > 0:
            print("Missing values found:")
            print(missing_values[missing_values > 0])
            
            # Fill missing values
            dataset.fillna(method='ffill', inplace=True)
            dataset.fillna(method='bfill', inplace=True)
            print("Missing values filled using forward and backward fill.")
        
        return dataset
        
    except Exception as e:
        print(f"Error loading Tesla data: {str(e)}")
        return None

def generate_model_report(model, model_name, metrics, save_path=None):
    """
    Generate a comprehensive model report.
    
    Args:
        model: Trained Keras model
        model_name (str): Name of the model
        metrics (dict): Dictionary of calculated metrics
        save_path (str): Path to save the report
    """
    try:
        report = []
        report.append("="*60)
        report.append(f"MODEL REPORT: {model_name.upper()}")
        report.append("="*60)
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Model Architecture
        report.append("MODEL ARCHITECTURE:")
        report.append("-" * 20)
        report.append(f"Total Parameters: {model.count_params():,}")
        report.append(f"Number of Layers: {len(model.layers)}")
        report.append("")
        
        # Performance Metrics
        if metrics:
            report.append("PERFORMANCE METRICS:")
            report.append("-" * 20)
            for metric, value in metrics.items():
                if isinstance(value, float):
                    if 'Accuracy' in metric or 'R_squared' in metric:
                        report.append(f"{metric}: {value:.4f}")
                    else:
                        report.append(f"{metric}: ${value:.2f}")
                else:
                    report.append(f"{metric}: {value}")
            report.append("")
        
        # Model Summary
        report.append("DETAILED MODEL SUMMARY:")
        report.append("-" * 25)
        
        # Convert report to string
        report_text = "\n".join(report)
        
        # Print report
        print(report_text)
        
        # Save report if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write(report_text)
                f.write("\n\nMODEL ARCHITECTURE DETAILS:\n")
                f.write("-" * 30 + "\n")
                model.summary(print_fn=lambda x: f.write(x + '\n'))
            print(f"Model report saved to {save_path}")
        
    except Exception as e:
        print(f"Error generating model report: {str(e)}")

def validate_data_quality(dataset, min_rows=100):
    """
    Validate the quality of the loaded dataset.
    
    Args:
        dataset (pd.DataFrame): Stock dataset to validate
        min_rows (int): Minimum number of rows required
    
    Returns:
        bool: True if data quality is acceptable, False otherwise
    """
    try:
        print("Validating data quality...")
        
        # Check if dataset exists and has minimum rows
        if dataset is None or len(dataset) < min_rows:
            print(f"Error: Dataset has insufficient data (minimum {min_rows} rows required)")
            return False
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
        missing_columns = [col for col in required_columns if col not in dataset.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            return False
        
        # Check for excessive missing values
        missing_percentage = (dataset.isnull().sum() / len(dataset)) * 100
        high_missing = missing_percentage[missing_percentage > 10]
        
        if not high_missing.empty:
            print(f"Warning: High missing values in columns: {high_missing.to_dict()}")
        
        # Check for data consistency
        inconsistent_data = dataset[dataset['High'] < dataset['Low']]
        if not inconsistent_data.empty:
            print(f"Warning: Found {len(inconsistent_data)} rows with High < Low")
        
        print("Data quality validation completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error validating data quality: {str(e)}")
        return False
