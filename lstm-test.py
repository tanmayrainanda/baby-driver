import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. DATA LOADING AND PREPROCESSING
def load_and_preprocess_data(file_path, manhattan_location_ids=None):
    """
    Load the taxi trip data and preprocess it for LSTM modeling
    """
    print("Loading taxi trip data...")
    # Define data types for optimized memory usage
    dtypes = {
        'VendorID': 'int16',
        'passenger_count': 'float32',
        'trip_distance': 'float32',
        'RatecodeID': 'float32',
        'store_and_fwd_flag': 'category',
        'PULocationID': 'int16',
        'DOLocationID': 'int16',
        'payment_type': 'int16',
        'fare_amount': 'float32',
        'extra': 'float32',
        'mta_tax': 'float32',
        'tip_amount': 'float32',
        'tolls_amount': 'float32',
        'improvement_surcharge': 'float32',
        'total_amount': 'float32',
        'congestion_surcharge': 'float32',
        'airport_fee': 'float32'
    }
    
    # Define datetime columns to parse
    datetime_columns = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']
    
    # Load the data - parquet doesn't support parse_dates parameter
    print(f"Reading parquet file from {file_path}...")
    try:
        # First try loading the file without specifying parse_dates (not supported in parquet)
        df = pd.read_parquet(file_path)
        print(f"Successfully loaded data with {df.shape[0]} rows")
        
        # Convert datetime columns after loading
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                
        # Apply dtypes after loading if needed
        for col, dtype in dtypes.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except Exception as e:
                    print(f"Warning: Could not convert column {col} to {dtype}: {e}")
    except Exception as e:
        print(f"Error loading parquet file: {e}")
        # Try reading CSV if parquet fails
        try:
            print("Attempting to read as CSV...")
            # For CSV, we can use parse_dates parameter
            df = pd.read_csv(file_path, dtype=dtypes, parse_dates=datetime_columns)
            print(f"Successfully loaded CSV data with {df.shape[0]} rows")
        except Exception as csv_e:
            print(f"Error loading CSV file: {csv_e}")
            raise RuntimeError("Could not load the dataset in either parquet or CSV format")
    
    # DEBUG: Display data statistics before filtering
    print("\n--- Data Statistics Before Filtering ---")
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['tpep_pickup_datetime'].min()} to {df['tpep_pickup_datetime'].max()}")
    print(f"Pickup location IDs: {df['PULocationID'].nunique()} unique values")
    
    # If manhattan_location_ids is None, try to load from lookup table
    if manhattan_location_ids is None:
        try:
            # Try to load the lookup table
            lookup_df = pd.read_csv("taxi_zone_lookup.csv")
            manhattan_lookup = lookup_df[lookup_df['Borough'] == 'Manhattan']
            manhattan_location_ids = manhattan_lookup['LocationID'].tolist()
            print(f"Identified {len(manhattan_location_ids)} Manhattan locations from lookup table")
        except Exception as e:
            print(f"Warning: Could not load borough information: {e}")
            print("Using all location IDs (will filter for Manhattan later if lookup table becomes available)")
    
    # Filter for Manhattan if location IDs are specified
    if manhattan_location_ids:
        # DEBUG: Check if manhattan_location_ids overlap with data
        overlap_ids = set(manhattan_location_ids).intersection(set(df['PULocationID'].unique()))
        print(f"Found {len(overlap_ids)} Manhattan location IDs in the dataset out of {len(manhattan_location_ids)} expected")
        
        if len(overlap_ids) == 0:
            print("WARNING: No Manhattan location IDs found in dataset! Using all locations.")
            manhattan_location_ids = df['PULocationID'].unique()
        else:
            df = df[df['PULocationID'].isin(manhattan_location_ids)]
            print(f"Filtered for Manhattan locations. Remaining rows: {df.shape[0]}")
    
    if manhattan_location_ids is None:
        # If we couldn't load the lookup table, get unique location IDs from the dataset
        unique_locations = df['PULocationID'].unique()
        print(f"Dataset contains {len(unique_locations)} unique pickup locations")
        manhattan_location_ids = unique_locations  # Use all locations for now
    
    print(f"Data loaded with {df.shape[0]} rows after filtering")
    
    # DEBUG: Verify we still have sufficient data after filtering
    if len(df) < 1000:
        print("WARNING: Very few records after filtering. This may cause training issues.")
    
    # Handle missing values
    print("Handling missing values...")
    for col in df.columns:
        if df[col].isna().sum() > 0:
            null_count = df[col].isna().sum()
            print(f"Column {col} has {null_count} null values ({null_count/len(df)*100:.2f}%)")
            
            if col in ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']:
                # For important numeric columns, use median
                df[col] = df[col].fillna(df[col].median())
            else:
                # For other columns, use mode
                df[col] = df[col].fillna(df[col].mode()[0])
    
    return df, manhattan_location_ids

def create_time_series_features(df):
    """
    Create time-based features for the LSTM model
    """
    print("Creating time-based features...")
    # Extract datetime features
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day'] = df['tpep_pickup_datetime'].dt.day
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
    df['month'] = df['tpep_pickup_datetime'].dt.month
    df['year'] = df['tpep_pickup_datetime'].dt.year
    
    # Create time bins (hourly aggregation)
    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.floor('H')
    
    return df

def aggregate_demand(df, location_id=None, location_names=None):
    """
    Aggregate taxi demand by hour and location
    """
    print("Aggregating demand by hour...")
    
    # Filter by location if specified
    if location_id:
        df = df[df['PULocationID'] == location_id]
        if location_names and location_id in location_names:
            print(f"Focusing on location: {location_names[location_id]} (ID: {location_id})")
        else:
            print(f"Focusing on location ID: {location_id}")
    else:
        print("Aggregating demand across all Manhattan locations")
    
    # Group by hour and count trips
    hourly_demand = df.groupby('pickup_hour').size().reset_index(name='demand')
    
    # DEBUG: Check aggregation results
    print(f"Aggregated hourly data contains {len(hourly_demand)} time points")
    print(f"Demand range: {hourly_demand['demand'].min()} to {hourly_demand['demand'].max()}")
    print(f"Demand mean: {hourly_demand['demand'].mean():.2f}, std: {hourly_demand['demand'].std():.2f}")
    
    # Ensure we have continuous hourly data
    start_date = hourly_demand['pickup_hour'].min()
    end_date = hourly_demand['pickup_hour'].max()
    
    # Create continuous time index
    full_hours = pd.date_range(start=start_date, end=end_date, freq='H')
    continuous_hourly = pd.DataFrame({'pickup_hour': full_hours})
    
    # Merge with actual demand data
    hourly_demand = pd.merge(continuous_hourly, hourly_demand, on='pickup_hour', how='left')
    
    # Fill missing values with 0 demand
    null_count = hourly_demand['demand'].isna().sum()
    if null_count > 0:
        print(f"Filling {null_count} missing hourly values ({null_count/len(hourly_demand)*100:.2f}%)")
    
    hourly_demand['demand'] = hourly_demand['demand'].fillna(0)
    
    # Add time-based features
    hourly_demand['hour'] = hourly_demand['pickup_hour'].dt.hour
    hourly_demand['day'] = hourly_demand['pickup_hour'].dt.day
    hourly_demand['day_of_week'] = hourly_demand['pickup_hour'].dt.dayofweek
    hourly_demand['month'] = hourly_demand['pickup_hour'].dt.month
    hourly_demand['year'] = hourly_demand['pickup_hour'].dt.year
    
    # Add hour-of-day seasonality features using sine and cosine transformations
    hours_in_day = 24
    hourly_demand['hour_sin'] = np.sin(2 * np.pi * hourly_demand['hour'] / hours_in_day)
    hourly_demand['hour_cos'] = np.cos(2 * np.pi * hourly_demand['hour'] / hours_in_day)
    
    # Add day-of-week seasonality features
    days_in_week = 7
    hourly_demand['day_of_week_sin'] = np.sin(2 * np.pi * hourly_demand['day_of_week'] / days_in_week)
    hourly_demand['day_of_week_cos'] = np.cos(2 * np.pi * hourly_demand['day_of_week'] / days_in_week)
    
    # Add month-of-year seasonality features
    months_in_year = 12
    hourly_demand['month_sin'] = np.sin(2 * np.pi * hourly_demand['month'] / months_in_year)
    hourly_demand['month_cos'] = np.cos(2 * np.pi * hourly_demand['month'] / months_in_year)
    
    return hourly_demand

# 2. PYTORCH DATASET CREATION
class TaxiDemandDataset(Dataset):
    """
    PyTorch Dataset for taxi demand forecasting
    """
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

def create_sequences(data, target_col, seq_length=24, forecast_horizon=1):
    """
    Create sequences for LSTM model training
    """
    print(f"Creating sequences with length {seq_length} and horizon {forecast_horizon}...")
    
    # DEBUG: Check data before scaling
    print(f"Data shape before scaling: {data.shape}")
    print(f"Target column '{target_col}' stats - min: {data[target_col].min()}, max: {data[target_col].max()}, mean: {data[target_col].mean():.2f}")
    
    # If demand has no variation, we'll have zero loss
    if data[target_col].std() < 0.01:
        print("WARNING: Target variable has almost no variation! This will cause zero loss.")
        print(f"Target std dev: {data[target_col].std():.6f}")
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # DEBUG: Check scaled data
    print(f"Scaled data shape: {scaled_data.shape}")
    target_idx = data.columns.get_loc(target_col)
    print(f"Target column index: {target_idx}")
    
    # Extract just the target column after scaling to verify it has variation
    scaled_target = scaled_data[:, target_idx]
    print(f"Scaled target stats - min: {scaled_target.min():.6f}, max: {scaled_target.max():.6f}, mean: {scaled_target.mean():.6f}, std: {scaled_target.std():.6f}")
    
    X, y = [], []
    
    # Create sequences
    for i in range(len(scaled_data) - seq_length - forecast_horizon + 1):
        # Input sequence
        X.append(scaled_data[i:(i + seq_length)])
        
        # Target value(s)
        if forecast_horizon == 1:
            y.append(scaled_data[i + seq_length, target_idx])
        else:
            y.append(scaled_data[(i + seq_length):(i + seq_length + forecast_horizon), target_idx])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # DEBUG: Check sequences
    print(f"Created {len(X)} sequences")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Verify sequences have variation
    if len(X) > 0:
        seq_means = X.mean(axis=(1, 2))
        seq_std = X.std(axis=(1, 2))
        print(f"Sequence means - min: {seq_means.min():.6f}, max: {seq_means.max():.6f}")
        print(f"Sequence std - min: {seq_std.min():.6f}, max: {seq_std.max():.6f}")
        
        # Check target variation
        y_mean = y.mean()
        y_std = y.std()
        print(f"Target y - mean: {y_mean:.6f}, std: {y_std:.6f}")
        
        if y_std < 0.001:
            print("WARNING: Target values have extremely low variation. This will cause zero loss.")
    else:
        print("WARNING: No sequences created! This will cause training to fail.")
    
    return X, y, scaler

# 3. LSTM MODEL DEFINITION
class LSTMModel(nn.Module):
    """
    LSTM model for time series forecasting
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: batch_size, seq_length, hidden_size
        
        # Get the output from the last time step
        out = out[:, -1, :]
        
        # Apply dropout
        out = self.dropout(out)
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        
        return out

# 4. TRAINING AND EVALUATION
def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=50, early_stopping_patience=10):
    """
    Train the LSTM model
    """
    print("Training LSTM model...")
    model.to(device)
    
    # For early stopping
    best_val_loss = float('inf')
    no_improve_epochs = 0
    best_model_state = None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': []
    }
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            # DEBUG: Check tensor shapes and values in first epoch
            if epoch == 0 and batch_count == 0:
                print(f"Input tensor shape: {sequences.shape}")
                print(f"Target tensor shape: {targets.shape}")
                print(f"Input tensor stats - min: {sequences.min().item():.6f}, max: {sequences.max().item():.6f}")
                print(f"Target tensor stats - min: {targets.min().item():.6f}, max: {targets.max().item():.6f}")
            
            # Forward pass
            outputs = model(sequences)
            
            # DEBUG: Check output in first epoch
            if epoch == 0 and batch_count == 0:
                print(f"Output tensor shape: {outputs.shape}")
                print(f"Output tensor stats - min: {outputs.min().item():.6f}, max: {outputs.max().item():.6f}")
            
            # Ensure targets have correct dimensions for criterion
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            
            loss = criterion(outputs, targets)
            
            # DEBUG: Check loss value
            if epoch == 0 and batch_count == 0:
                print(f"Initial loss value: {loss.item():.6f}")
                
                # If loss is zero, this is a critical issue
                if loss.item() == 0:
                    print("CRITICAL: Zero loss in first batch! Debugging tensor values:")
                    print(f"First few targets: {targets[:5].cpu().numpy()}")
                    print(f"First few outputs: {outputs[:5].cpu().numpy()}")
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        train_loss /= batch_count
        history['train_loss'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                
                outputs = model(sequences)
                
                # Ensure targets have correct dimensions
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                val_batch_count += 1
        
        val_loss /= max(val_batch_count, 1)  # Prevent division by zero
        history['val_loss'].append(val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_epochs = 0
            best_model_state = model.state_dict().copy()
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                model.load_state_dict(best_model_state)
                break
    
    if no_improve_epochs < early_stopping_patience:
        print(f'Training completed with {epochs} epochs')
        model.load_state_dict(best_model_state)
    
    return model, history

def evaluate_model(model, test_loader, criterion, device, scaler, target_col_idx, data_shape):
    """
    Evaluate model performance on test data
    """
    print("Evaluating model performance...")
    model.eval()
    
    predictions = []
    actual = []
    test_loss = 0.0
    batch_count = 0
    
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            
            outputs = model(sequences)
            
            # Ensure targets have correct dimensions
            if targets.dim() == 1:
                targets = targets.unsqueeze(1)
            
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            batch_count += 1
            
            # Store predictions and actual values
            predictions.extend(outputs.cpu().numpy())
            actual.extend(targets.cpu().numpy())
    
    test_loss /= max(batch_count, 1)  # Prevent division by zero
    print(f'Test Loss: {test_loss:.6f}')
    
    # Convert to arrays
    predictions = np.array(predictions).reshape(-1, 1)
    actual = np.array(actual).reshape(-1, 1)
    
    # Inverse transform predictions and actual values
    predictions_inv = inverse_transform_predictions(predictions, scaler, target_col_idx, data_shape)
    actual_inv = inverse_transform_predictions(actual, scaler, target_col_idx, data_shape)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(actual_inv, predictions_inv))
    mae = mean_absolute_error(actual_inv, predictions_inv)
    
    # Avoid division by zero in MAPE calculation
    epsilon = 1e-10
    mape = np.mean(np.abs((actual_inv - predictions_inv) / (actual_inv + epsilon))) * 100
    
    print(f"Test RMSE: {rmse:.2f}")
    print(f"Test MAE: {mae:.2f}")
    print(f"Test MAPE: {mape:.2f}%")
    
    return {
        'test_loss': test_loss,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'predictions': predictions_inv,
        'actual': actual_inv
    }

def inverse_transform_predictions(predictions, scaler, target_col_idx, data_shape):
    """
    Inverse transform scaled predictions back to original scale
    """
    # Create dummy array with same shape as original data
    dummy = np.zeros((len(predictions), data_shape))
    
    # Put predictions in the target column
    dummy[:, target_col_idx] = predictions.reshape(-1)
    
    # Inverse transform
    return scaler.inverse_transform(dummy)[:, target_col_idx]

def plot_results(history, predictions, actual, title="Taxi Demand Forecast", location_name=None):
    """
    Plot training history and forecasting results
    """
    if location_name:
        title = f"Taxi Demand Forecast - {location_name}"
    
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Plot predictions
    plt.subplot(2, 1, 2)
    plt.plot(actual, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title(title)
    plt.ylabel('Taxi Demand')
    plt.xlabel('Time Step')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# 5. FORECASTING FUTURE DEMAND
def forecast_future(model, last_sequence, scaler, target_col_idx, data_shape, n_steps=24, device='cpu', location_name=None):
    """
    Forecast future demand based on the last known sequence
    """
    if location_name:
        print(f"Forecasting {n_steps} steps ahead for {location_name}...")
    else:
        print(f"Forecasting {n_steps} steps ahead...")
    
    model.eval()
    
    # Start with the last known sequence
    curr_seq = torch.tensor(last_sequence, dtype=torch.float32).unsqueeze(0).to(device)
    
    # List to store predictions
    predictions = []
    
    # Predict one step at a time
    for _ in range(n_steps):
        with torch.no_grad():
            # Get prediction
            pred = model(curr_seq)
            predictions.append(pred.item())
            
            # Update sequence for next prediction
            new_seq = curr_seq.clone()
            new_seq[0, :-1, :] = curr_seq[0, 1:, :]
            
            # Create a new last step with the prediction (only update target column)
            new_last_step = new_seq[0, -1, :].clone()
            new_last_step[target_col_idx] = pred.item()
            new_seq[0, -1, :] = new_last_step
            
            curr_seq = new_seq
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    dummy = np.zeros((len(predictions), data_shape))
    dummy[:, target_col_idx] = predictions.reshape(-1)
    
    return scaler.inverse_transform(dummy)[:, target_col_idx]

# 6. MAIN EXECUTION
def main():
    """
    Main execution function
    """
    print("Starting taxi demand forecasting for Manhattan taxi services...")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load the taxi lookup table to identify Manhattan location IDs
    try:
        lookup_df = pd.read_csv("taxi_zone_lookup.csv")
        # Filter for Manhattan locations
        manhattan_lookup = lookup_df[lookup_df['Borough'] == 'Manhattan']
        manhattan_ids = manhattan_lookup['LocationID'].tolist()
        print(f"Loaded {len(manhattan_ids)} Manhattan location IDs from lookup table")
    except Exception as e:
        print(f"Warning: Could not load lookup table: {e}")
        print("Will use all location IDs from the dataset")
        manhattan_ids = None  # Will be determined from the dataset
    
    # 1. Load and preprocess data
    file_path = "taxi-dataset.parquet"  # Replace with your actual file path
    df, manhattan_ids = load_and_preprocess_data(file_path, manhattan_ids)
    
    # Load lookup table for location names (if available)
    try:
        lookup_df = pd.read_csv("taxi_zone_lookup.csv")
        print("Successfully loaded taxi location lookup table")
        # Create a mapping of location IDs to zone names
        location_names = dict(zip(lookup_df['LocationID'], lookup_df['Zone']))
        print(f"Created mapping for {len(location_names)} locations")
    except Exception as e:
        print(f"Warning: Could not load taxi location lookup table: {e}")
        location_names = None
    
    # 2. Create time-based features
    df = create_time_series_features(df)
    
    # 3. Aggregate demand (for all Manhattan)
    hourly_demand = aggregate_demand(df, location_id=None, location_names=location_names)
    
    # Select features for modeling including the cyclical features
    features = ['demand', 'hour', 'day_of_week', 'month', 
                'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 
                'month_sin', 'month_cos']
    model_data = hourly_demand[features].copy()
    
    # DEBUG: Verify data before sequence creation
    print("\n--- Feature Statistics ---")
    print(model_data.describe())
    
    # 4. Create sequences
    seq_length = 24  # Look back 24 hours
    forecast_horizon = 1  # Predict 1 hour ahead
    
    X, y, scaler = create_sequences(model_data, 'demand', seq_length, forecast_horizon)
    
    # Check if we have sufficient data to proceed
    if len(X) == 0 or np.isnan(X).any() or np.isnan(y).any():
        print("ERROR: Insufficient or invalid data after sequence creation.")
        return None, None, None
    
    # 5. Split data
    # Adjust the years based on your actual data range
    years_in_data = hourly_demand['year'].unique()
    years_in_data.sort()
    
    print(f"Years in data: {years_in_data}")
    
    if len(years_in_data) < 3:
        print("WARNING: Less than 3 years of data found. Adjusting split strategy.")
        # Calculate total length
        total_len = len(hourly_demand)
        # Use 70% for training, 15% for validation, 15% for testing
        train_end = int(total_len * 0.7)
        val_end = train_end + int(total_len * 0.15)
        
        # Adjust for sequence length
        train_end = max(0, train_end - seq_length - forecast_horizon + 1)
        val_end = max(train_end, val_end - seq_length - forecast_horizon + 1)
    else:
        # If we have sufficient years, use the last two for validation and testing
        test_year = years_in_data[-1]
        val_year = years_in_data[-2]
        train_years = years_in_data[:-2]
        
        print(f"Using years {train_years} for training, {val_year} for validation, {test_year} for testing")
        
        # Get indices for each split
        train_indices = hourly_demand[hourly_demand['year'].isin(train_years)].index
        val_indices = hourly_demand[hourly_demand['year'] == val_year].index
        test_indices = hourly_demand[hourly_demand['year'] == test_year].index
        
        # Adjust indices for sequence data
        train_end = len(train_indices) - seq_length - forecast_horizon + 1
        val_end = train_end + len(val_indices) - seq_length - forecast_horizon + 1
    
    # Ensure we don't have negative indices
    train_end = max(0, train_end)
    val_end = max(train_end, val_end)
    
    # Split data
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    print(f"Train data shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")
    
    # Check if we have enough data after splitting
    if len(X_train) < 100 or len(X_val) < 10 or len(X_test) < 10:
        print("WARNING: Very small split sizes. This may affect training/evaluation.")
    
    # 6. Create PyTorch datasets and dataloaders
    train_dataset = TaxiDemandDataset(X_train, y_train)
    val_dataset = TaxiDemandDataset(X_val, y_val)
    test_dataset = TaxiDemandDataset(X_test, y_test)
    
    batch_size = min(32, len(train_dataset))  # Ensure batch size isn't larger than dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 7. Build model
    input_size = X_train.shape[2]  # Number of features
    hidden_size = 64
    num_layers = 2
    output_size = 1  # Forecasting one step ahead
    dropout = 0.2
    
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
        dropout=dropout
    )
    
    # Print model summary
    print("\n--- Model Summary ---")
    print(model)
    print(f"Input size: {input_size}, Hidden size: {hidden_size}, Output size: {output_size}")
    
    # 8. Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 9. Train model
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=50,
        early_stopping_patience=10
    )
    
    # 10. Evaluate model
    target_col_idx = model_data.columns.get_loc('demand')
    eval_results = evaluate_model(
        model=trained_model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        scaler=scaler,
        target_col_idx=target_col_idx,
        data_shape=model_data.shape[1]
    )
    
    # 11. Plot results
    # Get location name if available
    location_name = "Manhattan"  # Default
    if 'location_names' in locals() and location_names:
        if len(manhattan_ids) == 1 and manhattan_ids[0] in location_names:
            location_name = location_names[manhattan_ids[0]]
    
    plot_results(
        history=history,
        predictions=eval_results['predictions'],
        actual=eval_results['actual'],
        title="Taxi Demand Forecast",
        location_name=location_name
    )
    
    # 12. Forecast future demand
    last_sequence = X_test[-1]
    future_forecast = forecast_future(
        model=trained_model,
        last_sequence=last_sequence,
        scaler=scaler,
        target_col_idx=target_col_idx,
        data_shape=model_data.shape[1],
        n_steps=24*7,  # Forecast one week ahead
        device=device,
        location_name=location_name
    )
    
    print("Forecasting complete!")
    print(f"Next 24 hours forecast: {future_forecast[:24]}")
    
    return trained_model, eval_results, future_forecast

if __name__ == "__main__":
    main()