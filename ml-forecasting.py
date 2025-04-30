import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler


class SimpleTimeSeriesForecaster:
    """
    A straightforward time series forecaster using RandomForest.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Initialize the forecaster.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            random_state: Random state for reproducibility
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        self.feature_cols = None
        self.importance = None
    
    def load_taxi_data(self, file_path, sample_frac=0.1, max_rows=10000):
        """
        Load and prepare taxi data from a parquet file.
        
        Args:
            file_path: Path to the parquet file
            sample_frac: Fraction of data to sample
            max_rows: Maximum number of rows to use
            
        Returns:
            Prepared DataFrame
        """
        try:
            # Read with specific columns only to reduce memory usage
            essential_columns = [
                'tpep_pickup_datetime', 'tpep_dropoff_datetime',
                'trip_distance', 'fare_amount'
            ]
            
            print(f"Loading data from {file_path}...")
            
            try:
                # Try to read with essential columns
                df = pd.read_parquet(file_path, columns=essential_columns)
            except Exception as e:
                print(f"Error reading with columns specified: {str(e)}")
                # Fallback to reading whole file
                df = pd.read_parquet(file_path)
            
            print(f"Original data shape: {df.shape}")
            
            # Take a random sample to reduce memory footprint
            if sample_frac < 1.0:
                df = df.sample(frac=sample_frac, random_state=42)
            
            # Further limit to max_rows if needed
            if len(df) > max_rows:
                df = df.head(max_rows)
                
            print(f"Sampled data shape: {df.shape}")
            
            # Convert datetime columns
            df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
            if 'tpep_dropoff_datetime' in df.columns:
                df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
            
            # Create fare rate
            if 'fare_amount' in df.columns and 'trip_distance' in df.columns:
                # Add small value to avoid division by zero
                df['fare_rate'] = df['fare_amount'] / df['trip_distance'].replace(0, 0.1)
                
                # Remove extreme values
                q1 = df['fare_rate'].quantile(0.01)
                q3 = df['fare_rate'].quantile(0.99)
                df['fare_rate'] = df['fare_rate'].clip(q1, q3)
                
            # Extract datetime features
            df = self._extract_datetime_features(df, 'tpep_pickup_datetime')
            
            # Ensure data is sorted by datetime
            df = df.sort_values('tpep_pickup_datetime')
            
            return df
            
        except Exception as e:
            print(f"Error in data loading: {str(e)}")
            return self._create_dummy_data()
    
    def _extract_datetime_features(self, df, datetime_col):
        """
        Extract datetime features for time series prediction.
        
        Args:
            df: DataFrame
            datetime_col: Name of datetime column
            
        Returns:
            DataFrame with additional features
        """
        df_copy = df.copy()
        
        # Basic time components
        df_copy['hour'] = df_copy[datetime_col].dt.hour
        df_copy['day'] = df_copy[datetime_col].dt.day
        df_copy['weekday'] = df_copy[datetime_col].dt.dayofweek
        df_copy['month'] = df_copy[datetime_col].dt.month
        df_copy['year'] = df_copy[datetime_col].dt.year
        df_copy['is_weekend'] = df_copy['weekday'].isin([5, 6]).astype(int)
        
        # Cyclical encoding (sine/cosine) to handle periodicity
        df_copy['hour_sin'] = np.sin(2 * np.pi * df_copy['hour'] / 24)
        df_copy['hour_cos'] = np.cos(2 * np.pi * df_copy['hour'] / 24)
        df_copy['day_sin'] = np.sin(2 * np.pi * df_copy['day'] / 31)
        df_copy['day_cos'] = np.cos(2 * np.pi * df_copy['day'] / 31)
        df_copy['month_sin'] = np.sin(2 * np.pi * df_copy['month'] / 12)
        df_copy['month_cos'] = np.cos(2 * np.pi * df_copy['month'] / 12)
        df_copy['weekday_sin'] = np.sin(2 * np.pi * df_copy['weekday'] / 7)
        df_copy['weekday_cos'] = np.cos(2 * np.pi * df_copy['weekday'] / 7)
        
        return df_copy
    
    def _create_dummy_data(self):
        """Create a small dummy dataset for testing."""
        print("Creating a dummy dataset for testing...")
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='H')
        
        # Create a DataFrame with synthetic time series data
        dummy_df = pd.DataFrame({
            'tpep_pickup_datetime': dates,
            'trip_distance': np.random.uniform(1, 10, size=1000),
            'fare_amount': np.random.uniform(5, 50, size=1000)
        })
        
        # Add some noise and trend
        t = np.arange(len(dummy_df))
        trend = 0.01 * t
        seasonality_day = 5 * np.sin(2 * np.pi * t / 24)  # Daily pattern
        seasonality_week = 10 * np.sin(2 * np.pi * t / (24 * 7))  # Weekly pattern
        noise = np.random.normal(0, 2, size=len(dummy_df))
        
        dummy_df['fare_rate'] = 2.5 + trend + seasonality_day + seasonality_week + noise
        dummy_df = self._extract_datetime_features(dummy_df, 'tpep_pickup_datetime')
        
        return dummy_df
    
    def create_time_series_features(self, df, target_col, 
                                   date_col='tpep_pickup_datetime',
                                   lag_periods=[1, 2, 3, 7, 14], 
                                   window_sizes=[3, 7, 14]):
        """
        Create time series features including lags and rolling windows.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            date_col: Date column name
            lag_periods: List of lag periods to create
            window_sizes: List of rolling window sizes
            
        Returns:
            DataFrame with time series features
        """
        # Create a copy and ensure it's sorted by date
        df_copy = df.copy()
        df_copy = df_copy.sort_values(date_col)
        
        # Create lag features
        for lag in lag_periods:
            df_copy[f'{target_col}_lag_{lag}'] = df_copy[target_col].shift(lag)
            
        # Create rolling window features
        for window in window_sizes:
            # Calculate rolling statistics
            df_copy[f'{target_col}_roll_mean_{window}'] = df_copy[target_col].rolling(window=window).mean()
            df_copy[f'{target_col}_roll_std_{window}'] = df_copy[target_col].rolling(window=window).std()
            
            # Calculate rate of change
            df_copy[f'{target_col}_roc_{window}'] = df_copy[target_col].pct_change(periods=window)
        
        return df_copy
    
    def prepare_train_test_data(self, df, target_col, test_size=0.2, 
                              feature_cols=None, create_features=True,
                              lag_periods=[1, 2, 3, 7], window_sizes=[3, 7]):
        """
        Prepare train/test data for time series forecasting.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            test_size: Proportion of data to use for testing
            feature_cols: List of feature columns (if None, use all except target)
            create_features: Whether to create lag and window features
            lag_periods: Lag periods to use if creating features
            window_sizes: Window sizes to use if creating features
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        # Create time series features if requested
        if create_features:
            df = self.create_time_series_features(
                df, target_col, lag_periods=lag_periods, window_sizes=window_sizes
            )
        
        # Sort by date
        if 'tpep_pickup_datetime' in df.columns:
            df = df.sort_values('tpep_pickup_datetime')
        
        # Drop rows with NaN (from lagged features)
        df = df.dropna()
        
        # Determine split point
        split_idx = int(len(df) * (1 - test_size))
        
        # Split data
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        # Determine feature columns
        if feature_cols is None:
            # Use all columns except target and date columns
            self.feature_cols = [col for col in df.columns 
                                if col != target_col 
                                and not col.startswith('tpep_')
                                and not pd.api.types.is_datetime64_any_dtype(df[col])]
        else:
            self.feature_cols = feature_cols
        
        # Create train/test sets
        X_train = train_df[self.feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[self.feature_cols]
        y_test = test_df[target_col]
        
        return X_train, X_test, y_train, y_test
    
    def train_evaluate(self, X_train, X_test, y_train, y_test):
        """
        Train the model and evaluate its performance.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"Training RandomForest model with {X_train.shape[0]} samples and {X_train.shape[1]} features")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))
        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)
        
        # Calculate MAPE
        def mape(y_true, y_pred):
            # Avoid division by zero
            mask = y_true != 0
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        train_mape = mape(y_train.values, train_preds)
        test_mape = mape(y_test.values, test_preds)
        
        # Store feature importance
        self.importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Print results
        print(f"\nTraining Results:")
        print(f"RMSE: {train_rmse:.4f}")
        print(f"MAE: {train_mae:.4f}")
        print(f"R²: {train_r2:.4f}")
        print(f"MAPE: {train_mape:.2f}%")
        
        print(f"\nTest Results:")
        print(f"RMSE: {test_rmse:.4f}")
        print(f"MAE: {test_mae:.4f}")
        print(f"R²: {test_r2:.4f}")
        print(f"MAPE: {test_mape:.2f}%")
        
        # Return evaluation results
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mape': train_mape,
            'test_mape': test_mape,
            'predictions': {
                'train': train_preds,
                'test': test_preds
            }
        }
    
    def cross_validation(self, X, y, n_splits=5, gap=0):
        """
        Perform time series cross-validation.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            n_splits: Number of CV splits
            gap: Gap between train and test sets
            
        Returns:
            Dictionary with CV results
        """
        # Create TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
        
        # Store results
        cv_scores = []
        cv_predictions = []
        
        # Perform CV
        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Make predictions
            preds = self.model.predict(X_test)
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            cv_scores.append(rmse)
            
            # Store predictions
            cv_predictions.append({
                'fold': i,
                'y_true': y_test,
                'y_pred': preds
            })
            
            print(f"Fold {i+1} RMSE: {rmse:.4f}")
        
        # Print average score
        print(f"Average CV RMSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        return {
            'cv_scores': cv_scores,
            'mean_score': np.mean(cv_scores),
            'std_score': np.std(cv_scores),
            'predictions': cv_predictions
        }
    
    def plot_feature_importance(self, top_n=20):
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to plot
        """
        if self.importance is None:
            print("No feature importance available. Train the model first.")
            return
        
        # Get top N features
        top_features = self.importance.head(top_n)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importance')
        plt.gca().invert_yaxis()  # Highest importance at the top
        plt.tight_layout()
        plt.show()
    
    def plot_actual_vs_predicted(self, y_true, y_pred, title='Actual vs Predicted'):
        """
        Plot actual vs predicted values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        
        # Convert to numpy arrays to be safe
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)
        
        # Create indices for x-axis
        indices = np.arange(len(y_true_array))
        
        # Plot actual and predicted
        plt.plot(indices, y_true_array, 'b-', label='Actual', alpha=0.7)
        plt.plot(indices, y_pred_array, 'r--', label='Predicted')
        
        plt.title(title)
        plt.xlabel('Time Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # Also create a scatter plot
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true_array, y_pred_array, alpha=0.5)
        plt.plot([y_true_array.min(), y_true_array.max()], 
                 [y_true_array.min(), y_true_array.max()], 
                 'k--', lw=2)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Prediction Scatter Plot')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main():
    """Main function to demonstrate RandomForest time series forecasting."""
    
    # Initialize the forecaster
    forecaster = SimpleTimeSeriesForecaster(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    
    # Load and prepare data
    # Replace this path with your actual file path
    df = forecaster.load_taxi_data('taxi-dataset.parquet', sample_frac=0.1)
    
    # Create time series features
    df = forecaster.create_time_series_features(df, 'fare_rate')
    
    # Prepare train/test data
    X_train, X_test, y_train, y_test = forecaster.prepare_train_test_data(
        df, 
        target_col='fare_rate',
        test_size=0.2,
        create_features=False  # Already created above
    )
    
    # Train and evaluate the model
    results = forecaster.train_evaluate(X_train, X_test, y_train, y_test)
    
    # Plot feature importance
    forecaster.plot_feature_importance(top_n=15)
    
    # Plot actual vs predicted
    forecaster.plot_actual_vs_predicted(
        y_test, 
        results['predictions']['test'],
        title='Test Set: Actual vs Predicted Fare Rate'
    )
    
    # Perform cross-validation
    cv_results = forecaster.cross_validation(
        pd.concat([X_train, X_test]),
        pd.concat([y_train, y_test]),
        n_splits=5
    )
    
    print("Analysis complete!")


if __name__ == "__main__":
    main()