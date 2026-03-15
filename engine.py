"""Enhanced stock price prediction pipeline for any ticker.
This module orchestrates the entire prediction workflow with improved error handling and logging.
Supports any stock ticker available on Yahoo Finance.
"""

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Import necessary modules and functions
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
import time

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import custom modules
from ml_pipeline.utils import (
    train_test_split, split_sequence, calculate_metrics, plot_predictions,
    process_and_split_multivariate_data, validate_data_quality,
    generate_model_report, save_predictions_to_csv, plot_technical_indicators,
    add_technical_indicators
)
from ml_pipeline.train import (
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

def load_stock_data_with_retry(ticker='AAPL', start_date='2010-01-01', end_date=None, max_retries=5):
    """
    Load stock data from Yahoo Finance with intelligent retry logic.
    Prioritizes REAL data over fallbacks.
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        start_date (str): Start date in YYYY-MM-DD format
        end_date (datetime): End date for data loading
        max_retries (int): Number of retry attempts
    
    Returns:
        pd.DataFrame: Real stock data from Yahoo Finance
    
    Raises:
        Exception: If unable to retrieve data after all retries
    """
    if end_date is None:
        end_date = datetime.now()
    
    print(f"\n{'='*70}")
    print(f"📊 FETCHING REAL DATA FOR {ticker} FROM YAHOO FINANCE")
    print(f"{'='*70}")
    print(f"Date Range: {start_date} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Attempts: 1 to {max_retries} (with increasing delays)\n")
    
    errors_log = []
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"🔄 Attempt {attempt}/{max_retries}...", end=" ", flush=True)
            
            # Fetch data directly from yfinance (more reliable than pdr override path)
            dataset = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=False,
                progress=False,
                threads=False
            )
            
            # Validate data
            if dataset is None:
                raise ValueError("DataFrame returned as None")
            
            if len(dataset) == 0:
                raise ValueError("Empty dataset returned - no data available for this date range")

            # Flatten MultiIndex columns (new yfinance behavior returns ('Close','AAPL') tuples)
            if isinstance(dataset.columns, pd.MultiIndex):
                dataset.columns = dataset.columns.get_level_values(0)

            # Success!
            print("✅ SUCCESS\n")
            print(f"✓ Data loaded: {len(dataset):,} trading days")
            print(f"✓ Date range: {dataset.index[0].date()} to {dataset.index[-1].date()}")
            print(f"✓ Columns: {', '.join(str(c) for c in dataset.columns)}")
            
            # Check for missing values
            missing_count = dataset.isnull().sum().sum()
            if missing_count > 0:
                print(f"⚠️  Found {missing_count} missing values - filling with forward/backward fill...")
                dataset.fillna(method='ffill', inplace=True)
                dataset.fillna(method='bfill', inplace=True)
            else:
                print(f"✓ No missing values")
            
            print(f"{'='*70}\n")
            return dataset
            
        except Exception as e:
            error_msg = str(e)
            errors_log.append(f"  Attempt {attempt}: {error_msg[:80]}")
            print(f"❌ FAILED")
            print(f"     Error: {error_msg[:60]}")
            
            if attempt < max_retries:
                # Exponential backoff: 1s, 2s, 4s, 8s, 16s
                wait_time = 2 ** (attempt - 1)
                print(f"     ⏳ Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...\n")
                time.sleep(wait_time)
            else:
                print(f"\n❌ FAILED AFTER {max_retries} ATTEMPTS\n")
    
    # All retries exhausted
    print("="*70)
    print("⚠️  UNABLE TO RETRIEVE REAL DATA FROM YAHOO FINANCE")
    print("="*70)
    print(f"Ticker: {ticker}")
    print(f"Date Range: {start_date} to {end_date.strftime('%Y-%m-%d')}")
    print(f"\n📋 Error Log:")
    for error in errors_log:
        print(error)
    print("\n💡 Troubleshooting:")
    print("   1. Check your internet connection")
    print("   2. Verify the ticker symbol is correct (e.g., AAPL, TSLA, MSFT)")
    print("   3. Try a different ticker to test connectivity")
    print("   4. Yahoo Finance API might be temporarily unavailable")
    print("   5. Try manually: https://finance.yahoo.com/quote/" + ticker)
    print("\n⚠️  CRITICAL: Aborting pipeline - synthetic data is NOT acceptable for this project")
    print("="*70 + "\n")
    
    raise Exception(f"Failed to retrieve real data for {ticker} from Yahoo Finance after {max_retries} attempts")

def main(ticker='AAPL', start_date='2010-01-01', end_date=None):
    """
    Main function to execute the stock prediction pipeline.
    
    Args:
        ticker (str): Stock ticker symbol (default: AAPL)
        start_date (str): Start date for data loading (default: 2010-01-01)
        end_date (str): End date for data loading (default: today)
    """
    if end_date is None:
        end_date_obj = datetime.now()
    else:
        try:
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        except:
            end_date_obj = datetime.now()
    
    print("="*80)
    print(f"STOCK PRICE PREDICTION PIPELINE - {ticker}")
    print("Enhanced RNN & LSTM Models with Technical Indicators")
    print("="*80)
    
    try:
        # Create output directories
        os.makedirs('output', exist_ok=True)
        os.makedirs('output/models', exist_ok=True)
        os.makedirs('output/figures', exist_ok=True)
        os.makedirs('output/reports', exist_ok=True)
        
        # Step 1: Load stock data with retry and fallback
        print("\n" + "="*60)
        print(f"STEP 1: LOADING {ticker} STOCK DATA")
        print("="*60)
        
        try:
            dataset = load_stock_data_with_retry(ticker, start_date, end_date_obj)
        except Exception as e:
            print(f"\n❌ PIPELINE ABORTED: Could not retrieve real data")
            print(f"Reason: {str(e)}")
            print("\n💡 NEXT STEPS:")
            print("   1. Check your internet connection")
            print(f"   2. Verify '{ticker}' is a valid stock ticker")
            print("   3. Try: python engine.py --ticker AAPL  (known working ticker)")
            print("   4. Visit https://finance.yahoo.com to verify ticker symbol")
            return False
        
        if dataset is None or len(dataset) == 0:
            print("\n❌ PIPELINE ABORTED: No real data available")
            return False
        
        # Validate data quality
        if not validate_data_quality(dataset):
            print("Data quality validation failed. Exiting...")
            return False
        
        # Create a checkpoint for tracking progress
        checkpoint(f'{ticker.lower()}_data_loaded')
        print(f"✓ {ticker} data loaded and validated successfully!")
        
        # Step 2: Data preprocessing and splitting
        print("\n" + "="*60)
        print("STEP 2: DATA PREPROCESSING AND SPLITTING")
        print("="*60)
        
        # Adjust time periods based on available data
        available_years = (dataset.index[-1] - dataset.index[0]).days / 365
        if available_years >= 10:
            tstart = 2017
            tend = 2022
        else:
            start_year = dataset.index[0].year
            end_year = dataset.index[-1].year
            tstart = start_year
            tend = max(start_year, end_year - 1)
        
        print(f"Training period: {tstart} to {tend}")
        print(f"Test period: {tend+1} onwards")
        
        # Split the dataset into training and test sets
        training_set, test_set = train_test_split(dataset, tstart, tend)
        
        if training_set is None or test_set is None:
            print("Failed to split data. Exiting...")
            return False
        
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
            save_model_path=f"output/models/{ticker.lower()}_rnn_model.h5",
            ticker=ticker
        )
        
        if model_rnn is not None:
            print("✓ RNN model trained successfully!")
            model_snapshot(f"{ticker.lower()}_rnn_trained")
        else:
            print("✗ RNN model training failed!")
        
        # Step 5: Train LSTM Model
        print("\n" + "="*60)
        print("STEP 5: TRAINING ENHANCED LSTM MODEL")
        print("="*60)
        
        model_lstm = train_lstm_model(
            X_train, y_train, n_steps, features, sc, test_set, dataset,
            epochs=50, batch_size=32, verbose=1, steps_in_future=30,
            save_model_path=f"output/models/{ticker.lower()}_lstm_model.h5",
            ticker=ticker
        )
        
        if model_lstm is not None:
            print("✓ LSTM model trained successfully!")
            model_snapshot(f"{ticker.lower()}_lstm_trained")
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
            dataset, tstart, tend, mv_features, ticker=ticker
        )
        
        if X_train_mv is not None:
            # Create sequences for multivariate data
            from ml_pipeline.utils import create_sequences_multivariate
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
                save_model_path=f"output/models/{ticker.lower()}_multivariate_lstm_model.h5",
                ticker=ticker
            )
            
            if model_mv is not None:
                print("✓ Multivariate LSTM model trained successfully!")
                model_snapshot(f"{ticker.lower()}_multivariate_lstm_trained")
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
            comparison_df = compare_models(trained_models, {}, ticker=ticker)
            
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
                multi_variate_df = add_technical_indicators(dataset)
                
                plot_technical_indicators(
                    multi_variate_df, tstart, tend,
                    save_path=f"output/figures/{ticker.lower()}_technical_indicators.png",
                    ticker=ticker
                )
                print("✓ Technical indicators visualization generated!")
            
        except Exception as e:
            print(f"✗ Error generating visualizations: {str(e)}")
        
        # Step 9: Final summary and cleanup
        print("\n" + "="*80)
        print(f"STOCK PRICE PREDICTION PIPELINE COMPLETED - {ticker}")
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
            model_file = f"{ticker.lower()}_{model_name.lower()}_model.h5"
            print(f"    - output/models/{model_file}")
        
        print("  Figures:")
        print(f"    - output/figures/{ticker.lower()}_technical_indicators.png")
        print(f"    - output/figures/*_predictions.png")
        
        print("  Reports:")
        print(f"    - output/{ticker.lower()}_model_comparison.csv")
        
        print("\n🎯 NEXT STEPS:")
        print("1. Review model performance metrics in the output files")
        print("2. Analyze prediction plots for model accuracy")
        print(f"3. Use the trained models for future {ticker} stock predictions")
        print("4. Consider fine-tuning hyperparameters for better performance")
        
        print("\n" + "="*80)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY! 🚀")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ PIPELINE EXECUTION FAILED: {str(e)}")
        print("Please check the error details above and try again.")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Stock Price Prediction Pipeline")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Stock ticker symbol (default: AAPL)")
    parser.add_argument("--start", type=str, default="2010-01-01", help="Start date for data loading (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="End date for data loading (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Execute the main pipeline
    success = main(ticker=args.ticker, start_date=args.start, end_date=args.end)
    
    if success:
        print(f"\n🎉 {args.ticker} Stock Prediction Pipeline executed successfully!")
    else:
        print(f"\n💥 Pipeline execution failed. Please check the logs above.")
