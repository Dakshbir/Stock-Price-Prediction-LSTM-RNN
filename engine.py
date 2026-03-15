"""
Enhanced main engine for Tesla stock price prediction pipeline.
This module orchestrates the entire prediction workflow with improved error handling and logging.
"""

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Import necessary modules and functions
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import tensorflow as tf

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from utils import (
    train_test_split, split_sequence, calculate_metrics, plot_predictions,
    process_and_split_multivariate_data, load_tesla_data, validate_data_quality,
    generate_model_report, save_predictions_to_csv, plot_technical_indicators
)
from train import (
    train_rnn_model, train_lstm_model, train_multivariate_lstm,
    compare_models, plot_training_history
)

# Import required libraries
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from pandas_datareader import data as pdr
from projectpro import model_snapshot, checkpoint

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Override Yahoo Finance's download method
yf.pdr_override()

def main():
    """
    Main function to execute the Tesla stock prediction pipeline.
    """
    print("="*80)
    print("TESLA STOCK PRICE PREDICTION PIPELINE")
    print("Enhanced RNN & LSTM Models with Technical Indicators")
    print("="*80)
    
    try:
        # Create output directories
        os.makedirs('output', exist_ok=True)
        os.makedirs('output/models', exist_ok=True)
        os.makedirs('output/figures', exist_ok=True)
        os.makedirs('output/reports', exist_ok=True)
        
        # Step 1: Load Tesla stock data
        print("\n" + "="*60)
        print("STEP 1: LOADING TESLA STOCK DATA")
        print("="*60)
        
        dataset = load_tesla_data(start_date='2012-01-01', end_date=datetime.now())
        
        if dataset is None:
            print("Failed to load Tesla data. Exiting...")
            return
        
        # Validate data quality
        if not validate_data_quality(dataset):
            print("Data quality validation failed. Exiting...")
            return
        
        # Create a checkpoint for tracking progress
        checkpoint('tesla_data_loaded')
        print("✓ Tesla data loaded and validated successfully!")
        
        # Step 2: Data preprocessing and splitting
        print("\n" + "="*60)
        print("STEP 2: DATA PREPROCESSING AND SPLITTING")
        print("="*60)
        
        # Set the start and end years for data splitting (adjusted for Tesla)
        tstart = 2017  # Tesla went public in 2010, but using 2017 for more stable data
        tend = 2022    # End year for training
        
        print(f"Training period: {tstart} to {tend}")
        print(f"Test period: {tend+1} onwards")
        
        # Split the dataset into training and test sets
        training_set, test_set = train_test_split(dataset, tstart, tend)
        
        if training_set is None or test_set is None:
            print("Failed to split data. Exiting...")
            return
        
        # Scale dataset values using Min-Max scaling
        sc = MinMaxScaler(feature_range=(0, 1))
        training_set = training_set.reshape(-1, 1)
        training_set_scaled = sc.fit_transform(training_set)
        
        print("✓ Data preprocessing completed successfully!")
        
        # Step 3: Create sequences for time series modeling
        print("\n" + "="*60)
        print("STEP 3: CREATING TIME SERIES SEQUENCES")
        print("="*60)
        
        # Create overlapping window batches (increased from 1 to 60 for better patterns)
        n_steps = 60
        features = 1
        
        X_train, y_train = split_sequence(training_set_scaled, n_steps)
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], features)
        
        print(f"✓ Training sequences created: {X_train.shape}")
        print(f"✓ Training targets created: {y_train.shape}")
        
        # Step 4: Train RNN Model
        print("\n" + "="*60)
        print("STEP 4: TRAINING ENHANCED RNN MODEL")
        print("="*60)
        
        model_rnn = train_rnn_model(
            X_train, y_train, n_steps, features, sc, test_set, dataset,
            epochs=50, batch_size=32, verbose=1, steps_in_future=30,
            save_model_path="output/models/tesla_rnn_model.h5"
        )
        
        if model_rnn is not None:
            print("✓ RNN model trained successfully!")
            model_snapshot("tesla_rnn_trained")
        else:
            print("✗ RNN model training failed!")
        
        # Step 5: Train LSTM Model
        print("\n" + "="*60)
        print("STEP 5: TRAINING ENHANCED LSTM MODEL")
        print("="*60)
        
        model_lstm = train_lstm_model(
            X_train, y_train, n_steps, features, sc, test_set, dataset,
            epochs=50, batch_size=32, verbose=1, steps_in_future=30,
            save_model_path="output/models/tesla_lstm_model.h5"
        )
        
        if model_lstm is not None:
            print("✓ LSTM model trained successfully!")
            model_snapshot("tesla_lstm_trained")
        else:
            print("✗ LSTM model training failed!")
        
        # Step 6: Prepare and train multivariate LSTM model
        print("\n" + "="*60)
        print("STEP 6: TRAINING MULTIVARIATE LSTM MODEL")
        print("="*60)
        
        # Set the number of multivariate features
        mv_features = 8
        
        # Process and split multivariate data
        X_train_mv, y_train_mv, X_test_mv, y_test_mv, mv_sc = process_and_split_multivariate_data(
            dataset, tstart, tend, mv_features
        )
        
        if X_train_mv is not None:
            # Create sequences for multivariate data
            from utils import create_sequences_multivariate
            mv_n_steps = 30
            
            X_train_mv_seq, y_train_mv_seq = create_sequences_multivariate(
                X_train_mv, y_train_mv, mv_n_steps
            )
            X_test_mv_seq, y_test_mv_seq = create_sequences_multivariate(
                X_test_mv, y_test_mv, mv_n_steps
            )
            
            print(f"✓ Multivariate sequences created:")
            print(f"  Training: {X_train_mv_seq.shape}")
            print(f"  Test: {X_test_mv_seq.shape}")
            
            # Train the multivariate LSTM model
            model_mv = train_multivariate_lstm(
                X_train_mv_seq, y_train_mv_seq, X_test_mv_seq, y_test_mv_seq,
                mv_features, mv_sc, epochs=50, batch_size=32, verbose=1,
                save_model_path="output/models/tesla_multivariate_lstm_model.h5"
            )
            
            if model_mv is not None:
                print("✓ Multivariate LSTM model trained successfully!")
                model_snapshot("tesla_multivariate_lstm_trained")
            else:
                print("✗ Multivariate LSTM model training failed!")
        else:
            print("✗ Multivariate data processing failed!")
            model_mv = None
        
        # Step 7: Model comparison and final evaluation
        print("\n" + "="*60)
        print("STEP 7: MODEL COMPARISON AND EVALUATION")
        print("="*60)
        
        # Collect trained models
        trained_models = {}
        if model_rnn is not None:
            trained_models['RNN'] = model_rnn
        if model_lstm is not None:
            trained_models['LSTM'] = model_lstm
        if model_mv is not None:
            trained_models['Multivariate_LSTM'] = model_mv
        
        if trained_models:
            print(f"✓ Successfully trained {len(trained_models)} models:")
            for model_name in trained_models.keys():
                print(f"  - {model_name}")
            
            # Generate comparison report
            comparison_df = compare_models(trained_models, {})
            
            if comparison_df is not None:
                print("✓ Model comparison completed!")
        else:
            print("✗ No models were successfully trained!")
        
        # Step 8: Generate technical indicators visualization
        print("\n" + "="*60)
        print("STEP 8: GENERATING VISUALIZATIONS")
        print("="*60)
        
        try:
            # Plot technical indicators if multivariate data was processed
            if X_train_mv is not None:
                # Recreate the multivariate dataframe for plotting
                multi_variate_df = dataset.copy()
                
                # Add technical indicators
                import pandas_ta as ta
                multi_variate_df['RSI'] = ta.rsi(multi_variate_df.Close, length=14)
                multi_variate_df['EMA_12'] = ta.ema(multi_variate_df.Close, length=12)
                multi_variate_df['EMA_26'] = ta.ema(multi_variate_df.Close, length=26)
                multi_variate_df['EMA_50'] = ta.ema(multi_variate_df.Close, length=50)
                
                # MACD
                macd_data = ta.macd(multi_variate_df.Close)
                if 'MACD_12_26_9' in macd_data.columns:
                    multi_variate_df['MACD'] = macd_data['MACD_12_26_9']
                
                plot_technical_indicators(
                    multi_variate_df, tstart, tend,
                    save_path="output/figures/tesla_technical_indicators.png"
                )
                print("✓ Technical indicators visualization generated!")
            
        except Exception as e:
            print(f"✗ Error generating visualizations: {str(e)}")
        
        # Step 9: Final summary and cleanup
        print("\n" + "="*80)
        print("TESLA STOCK PREDICTION PIPELINE COMPLETED")
        print("="*80)
        
        print("\n📊 PIPELINE SUMMARY:")
        print(f"✓ Data loaded: {dataset.shape[0]} records from {dataset.index[0].date()} to {dataset.index[-1].date()}")
        print(f"✓ Training period: {tstart} to {tend}")
        print(f"✓ Test period: {tend+1} onwards")
        print(f"✓ Models trained: {len(trained_models)}")
        print(f"✓ Output files saved to: ./output/")
        
        print("\n📁 OUTPUT FILES:")
        print("  Models:")
        for model_name in trained_models.keys():
            model_file = f"tesla_{model_name.lower()}_model.h5"
            print(f"    - output/models/{model_file}")
        
        print("  Figures:")
        print("    - output/figures/tesla_technical_indicators.png")
        print("    - output/figures/*_predictions.png")
        
        print("  Reports:")
        print("    - output/tesla_model_comparison.csv")
        
        print("\n🎯 NEXT STEPS:")
        print("1. Review model performance metrics in the output files")
        print("2. Analyze prediction plots for model accuracy")
        print("3. Use the trained models for future Tesla stock predictions")
        print("4. Consider fine-tuning hyperparameters for better performance")
        
        print("\n" + "="*80)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY! 🚀")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ PIPELINE EXECUTION FAILED: {str(e)}")
        print("Please check the error details above and try again.")
        return False
    
    return True

if __name__ == "__main__":
    # Execute the main pipeline
    success = main()
    
    if success:
        print("\n🎉 Tesla Stock Prediction Pipeline executed successfully!")
    else:
        print("\n💥 Pipeline execution failed. Please check the logs above.")
