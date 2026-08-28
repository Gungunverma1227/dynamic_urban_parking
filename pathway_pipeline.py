import pathway as pw
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Optional
import json

class StreamingPricingState:
    """Manages state for streaming pricing pipeline"""
    
    def __init__(self):
        # State for baseline model
        self.last_prices = {}  # lot_id -> last_price
        
        # State for demand model
        self.rolling_windows = defaultdict(lambda: defaultdict(deque))  # lot_id -> feature -> deque
        self.raw_demand_history = defaultdict(deque)  # lot_id -> deque of raw demands
        self.normalized_demand_history = defaultdict(deque)  # lot_id -> deque of normalized demands
        self.lot_stats = defaultdict(lambda: {'min_demand': float('inf'), 'max_demand': float('-inf')})
        
        # State for competitive model
        self.current_prices = {}  # lot_id -> current_price (for competitive comparison)
        self.adjusted_demand_history = defaultdict(deque)  # lot_id -> deque of adjusted demands
        self.adjusted_demand_stats = defaultdict(lambda: {'min': float('inf'), 'max': float('-inf')})
        self.last_adjusted_prices = {}  # lot_id -> last_adjusted_price
        
        # State for nearby mapping (dynamic updates)
        self.nearby_map = {}  # lot_id -> list of nearby lots
        self.lots_coordinates = {}  # lot_id -> {'Latitude': x, 'Longitude': y}
        self.nearby_map_update_counter = 0
        self.nearby_map_update_frequency = 10  # Update every N records
        
        # Configuration
        self.window_size = 3
        self.max_history = 100  # Keep limited history for efficiency
        
    def update_lot_coordinates(self, lot_id, latitude, longitude):
        """Update coordinates for a lot"""
        self.lots_coordinates[lot_id] = {
            'Latitude': latitude,
            'Longitude': longitude
        }
        
    def should_update_nearby_map(self):
        """Check if we should update the nearby mapping"""
        self.nearby_map_update_counter += 1
        if self.nearby_map_update_counter >= self.nearby_map_update_frequency:
            self.nearby_map_update_counter = 0
            return True
        return False
        
    def update_nearby_map(self):
        """Update the nearby mapping based on current lots"""
        if len(self.lots_coordinates) < 2:
            return
            
        try:
            # Convert to DataFrame format
            lots_data = []
            for lot_id, coords in self.lots_coordinates.items():
                lots_data.append({
                    'SystemCodeNumber': lot_id,
                    'Latitude': coords['Latitude'],
                    'Longitude': coords['Longitude']
                })
            
            df_meta = pd.DataFrame(lots_data)
            self.nearby_map = compute_nearby_lots(df_meta, radius_km=1.0)
            print(f"Updated nearby mapping for {len(self.nearby_map)} lots")
            
        except Exception as e:
            print(f"Error updating nearby map: {e}")

# Global state instance
pricing_state = StreamingPricingState()

def haversine(lat1, lon1, lat2, lon2):
    """Calculate haversine distance between two points"""
    R = 6371  # Earth's radius in km
    
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def preprocess_row(system_code, date, time, lat, lon, capacity, occupancy, queue_length, vehicle_type, traffic, special_day):
    """Apply preprocessing transformations to individual columns"""
    try:
        # Create timestamp from date and time
        timestamp = pd.to_datetime(date + ' ' + time, format="%d-%m-%Y %H:%M:%S")
        
        # Calculate derived features
        occupancy_rate = occupancy / capacity if capacity > 0 else 0
        
        # Vehicle type mapping
        vehicle_type_map = {'car': 0.6, 'bike': 0.2, 'truck': 1}
        vehicle_type_weight = vehicle_type_map.get(vehicle_type, 0.6)
        
        # Traffic level mapping
        traffic_map = {'low': 0, 'average': 0.5, 'high': 1}
        traffic_level = traffic_map.get(traffic, 0.5)
        
        # Queue length normalization
        queue_length_norm = queue_length / 100.0  # Simple normalization
        
        # Update coordinates in state for nearby mapping
        pricing_state.update_lot_coordinates(system_code, lat, lon)
        
        # Update nearby mapping periodically
        if pricing_state.should_update_nearby_map():
            pricing_state.update_nearby_map()
        
        return (
            timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            occupancy_rate,
            queue_length_norm,
            vehicle_type_weight,
            traffic_level
        )
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return (
            str(datetime.now()),
            0,
            0,
            0.6,
            0.5
        )

def apply_baseline_model(system_code, occupancy_rate, alpha=2.0, threshold_occupancy=0.5, base_price=10.0, min_price=5.0, max_price=20.0):
    """Apply baseline linear model"""
    try:
        # Get previous price or use base price
        prev_price = pricing_state.last_prices.get(system_code, base_price)
        
        # Calculate new price
        new_price = prev_price + alpha * (occupancy_rate - threshold_occupancy)
        new_price = max(min(new_price, max_price), min_price)
        
        # Update state
        pricing_state.last_prices[system_code] = new_price
        
        return new_price
    except Exception as e:
        print(f"Error in baseline model: {e}")
        return base_price

def apply_demand_model(system_code, occupancy_rate, queue_length_norm, traffic_level, special_day, vehicle_type_weight, 
                      base_price=10.0, alpha=0.5, beta=0.2, gamma=0.15, delta=1.0, epsilon=0.5, 
                      lambda_scale=0.8, min_price=5, max_price=20, smoothing_factor=0.1):
    """Apply demand-based model"""
    try:
        # Features for smoothing
        features = {
            'OccupancyRate': occupancy_rate,
            'QueueLength': queue_length_norm,
            'TrafficLevel': traffic_level,
            'VehicleTypeWeight': vehicle_type_weight
        }
        
        smoothed_features = {}
        
        # Update rolling windows and calculate smoothed features
        for feat_name, feat_value in features.items():
            window = pricing_state.rolling_windows[system_code][feat_name]
            window.append(feat_value)
            if len(window) > pricing_state.window_size:
                window.popleft()
            smoothed_features[f'{feat_name}_s'] = np.mean(list(window))
        
        # Calculate raw demand
        raw_demand = (
            alpha * smoothed_features['OccupancyRate_s'] +
            beta * smoothed_features['QueueLength_s'] +
            gamma * smoothed_features['TrafficLevel_s'] +
            delta * special_day +
            epsilon * smoothed_features['VehicleTypeWeight_s']
        )
        
        # Update raw demand history
        pricing_state.raw_demand_history[system_code].append(raw_demand)
        if len(pricing_state.raw_demand_history[system_code]) > pricing_state.max_history:
            pricing_state.raw_demand_history[system_code].popleft()
        
        # Update min/max for normalization
        lot_demands = list(pricing_state.raw_demand_history[system_code])
        if lot_demands:
            pricing_state.lot_stats[system_code]['min_demand'] = min(lot_demands)
            pricing_state.lot_stats[system_code]['max_demand'] = max(lot_demands)
        
        # Normalize demand
        min_demand = pricing_state.lot_stats[system_code]['min_demand']
        max_demand = pricing_state.lot_stats[system_code]['max_demand']
        
        if max_demand > min_demand:
            normalized_demand = (raw_demand - min_demand) / (max_demand - min_demand + 1e-5) * 2 - 1
        else:
            normalized_demand = 0
        
        # Update normalized demand history for smoothing
        pricing_state.normalized_demand_history[system_code].append(normalized_demand)
        if len(pricing_state.normalized_demand_history[system_code]) > pricing_state.window_size:
            pricing_state.normalized_demand_history[system_code].popleft()
        
        # Calculate smoothed demand
        smoothed_demand = np.mean(list(pricing_state.normalized_demand_history[system_code]))
        
        # Calculate price
        prev_price = pricing_state.last_prices.get(system_code, base_price)
        delta_price = lambda_scale * smoothed_demand
        new_price = prev_price * (1 - smoothing_factor) + base_price * (1 + delta_price) * smoothing_factor
        new_price = min(max(new_price, min_price), max_price)
        
        # Update state
        pricing_state.last_prices[system_code] = new_price
        pricing_state.current_prices[system_code] = new_price
        
        return new_price, smoothed_demand, raw_demand, normalized_demand
        
    except Exception as e:
        print(f"Error in demand model: {e}")
        return base_price, 0, 0, 0

def apply_competitive_model(system_code, price, smoothed_demand, base_price=10, mu=0.3, smoothing_factor=0.1, 
                           lambda_scale=0.8, min_price=5, max_price=20):
    """Apply competitive model"""
    try:
        # Get nearby lots
        neighbors = pricing_state.nearby_map.get(system_code, [])
        
        # Calculate competitive adjustment
        adjusted_demand = smoothed_demand
        if neighbors:
            # Get current prices of nearby lots
            comp_prices = [pricing_state.current_prices.get(neighbor, base_price) 
                          for neighbor in neighbors 
                          if neighbor in pricing_state.current_prices]
            
            if comp_prices:
                comp_avg = np.mean(comp_prices)
                adjusted_demand -= mu * np.tanh(price - comp_avg)
        
        # Update adjusted demand history
        pricing_state.adjusted_demand_history[system_code].append(adjusted_demand)
        if len(pricing_state.adjusted_demand_history[system_code]) > pricing_state.max_history:
            pricing_state.adjusted_demand_history[system_code].popleft()
        
        # Update adjusted demand stats for normalization
        adj_demands = list(pricing_state.adjusted_demand_history[system_code])
        if adj_demands:
            pricing_state.adjusted_demand_stats[system_code]['min'] = min(adj_demands)
            pricing_state.adjusted_demand_stats[system_code]['max'] = max(adj_demands)
        
        # Normalize adjusted demand
        min_adj = pricing_state.adjusted_demand_stats[system_code]['min']
        max_adj = pricing_state.adjusted_demand_stats[system_code]['max']
        
        if max_adj > min_adj:
            adjusted_demand_norm = (adjusted_demand - min_adj) / (max_adj - min_adj + 1e-5) * 2 - 1
        else:
            adjusted_demand_norm = 0
        
        # Calculate adjusted price
        prev_price = pricing_state.last_adjusted_prices.get(system_code, base_price)
        delta_price = lambda_scale * adjusted_demand_norm
        new_price = prev_price * (1 - smoothing_factor) + base_price * (1 + delta_price) * smoothing_factor
        new_price = min(max(new_price, min_price), max_price)
        
        # Update state
        pricing_state.last_adjusted_prices[system_code] = new_price
        pricing_state.current_prices[system_code] = new_price  # Update for competitive comparison
        
        return new_price, adjusted_demand, adjusted_demand_norm
        
    except Exception as e:
        print(f"Error in competitive model: {e}")
        return base_price, 0, 0

def compute_nearby_lots(df_meta, radius_km=1.0):
    """Compute nearby lots mapping using haversine distance"""
    nearby_map = {}
    for i, row1 in df_meta.iterrows():
        lot = row1['SystemCodeNumber']
        nearby = []
        for j, row2 in df_meta.iterrows():
            if lot == row2['SystemCodeNumber']:
                continue
            d = haversine(row1['Latitude'], row1['Longitude'], row2['Latitude'], row2['Longitude'])
            if d <= radius_km:
                nearby.append(row2['SystemCodeNumber'])
        nearby_map[lot] = nearby
    return nearby_map

def initialize_from_historical_data(historical_file):
    """Initialize state from historical data"""
    try:
        # Load historical data to initialize state
        df = pd.read_csv(historical_file)
        
        # Initialize last prices from most recent data per lot
        for lot_id in df['SystemCodeNumber'].unique():
            lot_data = df[df['SystemCodeNumber'] == lot_id]
            if not lot_data.empty:
                # Get the most recent price (assuming you have a Price column)
                if 'Price' in lot_data.columns:
                    last_price = lot_data['Price'].iloc[-1]
                    pricing_state.last_prices[lot_id] = last_price
                    pricing_state.current_prices[lot_id] = last_price
        
        print(f"Initialized state for {len(pricing_state.last_prices)} lots")
    except Exception as e:
        print(f"Warning: Could not initialize from historical data: {e}")

def simulate_streaming_csv(input_file, output_file, delay_seconds=1):
    """
    Simulate streaming by copying CSV data line by line with delays
    This creates a streaming effect for testing purposes
    """
    import time
    import os
    
    # Read the input CSV
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    # Write header first
    with open(output_file, 'w') as outfile:
        outfile.write(lines[0])  # Write header
        outfile.flush()
    
    # Write data lines with delay
    for i, line in enumerate(lines[1:], 1):
        with open(output_file, 'a') as outfile:
            outfile.write(line)
            outfile.flush()
        
        print(f"Streamed record {i}/{len(lines)-1}")
        time.sleep(delay_seconds)
    
    print("Streaming simulation complete")

def create_streaming_pipeline(input_file="dataset.csv"):
    """Create the complete Pathway streaming pipeline"""
    
    # Define input schema based on your raw data structure
    input_schema = pw.schema_builder(
        columns={
            "SystemCodeNumber": pw.column_definition(dtype=str),
            "LastUpdatedDate": pw.column_definition(dtype=str),
            "LastUpdatedTime": pw.column_definition(dtype=str),
            "Latitude": pw.column_definition(dtype=float),
            "Longitude": pw.column_definition(dtype=float),
            "Capacity": pw.column_definition(dtype=int),
            "Occupancy": pw.column_definition(dtype=int),
            "QueueLength": pw.column_definition(dtype=int),
            "VehicleType": pw.column_definition(dtype=str),
            "TrafficConditionNearby": pw.column_definition(dtype=str),
            "IsSpecialDay": pw.column_definition(dtype=int),
        }
    )
    
    # Input stream with streaming simulation and delay
    input_stream = pw.io.csv.read(
        input_file,
        mode="streaming",
        schema=input_schema,
        autocommit_duration_ms=1000,  # Process data every 1 second
        with_metadata=True  # Include metadata for better streaming control
    )
    
    # Step 1: Preprocess data
    preprocessed = input_stream.select(
        SystemCodeNumber=pw.this.SystemCodeNumber,
        Latitude=pw.this.Latitude,
        Longitude=pw.this.Longitude,
        Capacity=pw.this.Capacity,
        Occupancy=pw.this.Occupancy,
        QueueLength=pw.this.QueueLength,
        IsSpecialDay=pw.this.IsSpecialDay,
        # Apply preprocessing function
        Timestamp=pw.apply(lambda sc, dt, tm, lat, lon, cap, occ, ql, vt, tc, sd: preprocess_row(
            sc, dt, tm, lat, lon, cap, occ, ql, vt, tc, sd
        )[0],
            pw.this.SystemCodeNumber, pw.this.LastUpdatedDate, pw.this.LastUpdatedTime, 
            pw.this.Latitude, pw.this.Longitude, pw.this.Capacity, pw.this.Occupancy, 
            pw.this.QueueLength, pw.this.VehicleType, pw.this.TrafficConditionNearby, 
            pw.this.IsSpecialDay
        ),
        OccupancyRate=pw.apply(lambda sc, dt, tm, lat, lon, cap, occ, ql, vt, tc, sd: preprocess_row(
            sc, dt, tm, lat, lon, cap, occ, ql, vt, tc, sd
        )[1],
            pw.this.SystemCodeNumber, pw.this.LastUpdatedDate, pw.this.LastUpdatedTime, 
            pw.this.Latitude, pw.this.Longitude, pw.this.Capacity, pw.this.Occupancy, 
            pw.this.QueueLength, pw.this.VehicleType, pw.this.TrafficConditionNearby, 
            pw.this.IsSpecialDay
        ),
        QueueLengthNorm=pw.apply(lambda sc, dt, tm, lat, lon, cap, occ, ql, vt, tc, sd: preprocess_row(
            sc, dt, tm, lat, lon, cap, occ, ql, vt, tc, sd
        )[2],
            pw.this.SystemCodeNumber, pw.this.LastUpdatedDate, pw.this.LastUpdatedTime, 
            pw.this.Latitude, pw.this.Longitude, pw.this.Capacity, pw.this.Occupancy, 
            pw.this.QueueLength, pw.this.VehicleType, pw.this.TrafficConditionNearby, 
            pw.this.IsSpecialDay
        ),
        VehicleTypeWeight=pw.apply(lambda sc, dt, tm, lat, lon, cap, occ, ql, vt, tc, sd: preprocess_row(
            sc, dt, tm, lat, lon, cap, occ, ql, vt, tc, sd
        )[3],
            pw.this.SystemCodeNumber, pw.this.LastUpdatedDate, pw.this.LastUpdatedTime, 
            pw.this.Latitude, pw.this.Longitude, pw.this.Capacity, pw.this.Occupancy, 
            pw.this.QueueLength, pw.this.VehicleType, pw.this.TrafficConditionNearby, 
            pw.this.IsSpecialDay
        ),
        TrafficLevel=pw.apply(lambda sc, dt, tm, lat, lon, cap, occ, ql, vt, tc, sd: preprocess_row(
            sc, dt, tm, lat, lon, cap, occ, ql, vt, tc, sd
        )[4],
            pw.this.SystemCodeNumber, pw.this.LastUpdatedDate, pw.this.LastUpdatedTime, 
            pw.this.Latitude, pw.this.Longitude, pw.this.Capacity, pw.this.Occupancy, 
            pw.this.QueueLength, pw.this.VehicleType, pw.this.TrafficConditionNearby, 
            pw.this.IsSpecialDay
        ),
    )
    
    # Step 2: Apply baseline model
    with_baseline = preprocessed.select(
        *pw.this,
        Price=pw.apply(apply_baseline_model, pw.this.SystemCodeNumber, pw.this.OccupancyRate)
    )
    
    # Step 3: Apply demand model
    with_demand = with_baseline.select(
        *pw.this,
        DemandPrice=pw.apply(lambda sc, or_, qln, tl, sd, vtw: apply_demand_model(
            sc, or_, qln, tl, sd, vtw
        )[0], pw.this.SystemCodeNumber, pw.this.OccupancyRate, pw.this.QueueLengthNorm, 
            pw.this.TrafficLevel, pw.this.IsSpecialDay, pw.this.VehicleTypeWeight),
        SmoothedDemand=pw.apply(lambda sc, or_, qln, tl, sd, vtw: apply_demand_model(
            sc, or_, qln, tl, sd, vtw
        )[1], pw.this.SystemCodeNumber, pw.this.OccupancyRate, pw.this.QueueLengthNorm, 
            pw.this.TrafficLevel, pw.this.IsSpecialDay, pw.this.VehicleTypeWeight),
        RawDemand=pw.apply(lambda sc, or_, qln, tl, sd, vtw: apply_demand_model(
            sc, or_, qln, tl, sd, vtw
        )[2], pw.this.SystemCodeNumber, pw.this.OccupancyRate, pw.this.QueueLengthNorm, 
            pw.this.TrafficLevel, pw.this.IsSpecialDay, pw.this.VehicleTypeWeight),
        NormalizedDemand=pw.apply(lambda sc, or_, qln, tl, sd, vtw: apply_demand_model(
            sc, or_, qln, tl, sd, vtw
        )[3], pw.this.SystemCodeNumber, pw.this.OccupancyRate, pw.this.QueueLengthNorm, 
            pw.this.TrafficLevel, pw.this.IsSpecialDay, pw.this.VehicleTypeWeight),
    )
    
    # Step 4: Apply competitive model
    final_pricing = with_demand.select(
        *pw.this,
        AdjustedPrice=pw.apply(lambda sc, p, sd: apply_competitive_model(
            sc, p, sd
        )[0], pw.this.SystemCodeNumber, pw.this.DemandPrice, pw.this.SmoothedDemand),
        AdjustedDemand=pw.apply(lambda sc, p, sd: apply_competitive_model(
            sc, p, sd
        )[1], pw.this.SystemCodeNumber, pw.this.DemandPrice, pw.this.SmoothedDemand),
        AdjustedDemandNorm=pw.apply(lambda sc, p, sd: apply_competitive_model(
            sc, p, sd
        )[2], pw.this.SystemCodeNumber, pw.this.DemandPrice, pw.this.SmoothedDemand),
    )
    
    # Output streams
    pw.io.csv.write(final_pricing, "streaming_pricing_output.csv")
    
    # Create a summary stream for monitoring
    summary = final_pricing.select(
        pw.this.SystemCodeNumber,
        pw.this.Timestamp,
        pw.this.Price,
        pw.this.DemandPrice,
        pw.this.AdjustedPrice,
        pw.this.SmoothedDemand,
        pw.this.OccupancyRate
    )
    
    pw.io.csv.write(summary, "pricing_summary.csv")
    
    return final_pricing

def main():
    """Main function to run the streaming pipeline"""
    import threading
    import time
    import os
    
    # Configuration
    original_csv = "dataset.csv"  # Your original CSV file
    streaming_csv = "streaming_parking_data.csv"  # Temporary streaming file
    delay_seconds = 2  # Delay between records
    
    # Validate input file exists
    if not os.path.exists(original_csv):
        print(f"Error: Input file '{original_csv}' not found!")
        print("Please update the 'original_csv' variable with your actual CSV file path.")
        return
    
    # Initialize state from historical data if available
    # initialize_from_historical_data('historical_pricing.csv')
    
    # Start streaming simulation in a separate thread
    def start_streaming():
        time.sleep(2)  # Give pipeline time to start
        simulate_streaming_csv(original_csv, streaming_csv, delay_seconds)
    
    streaming_thread = threading.Thread(target=start_streaming)
    streaming_thread.daemon = True
    streaming_thread.start()
    
    # Create and run the streaming pipeline
    final_pricing = create_streaming_pipeline(streaming_csv)
    
    # Run the pipeline
    print("Starting streaming pricing pipeline...")
    print(f"Simulating stream with {delay_seconds}s delay between records")
    print(f"Input file: {original_csv}")
    print("Output files: streaming_pricing_output.csv, pricing_summary.csv")
    print("Press Ctrl+C to stop the pipeline")
    
    try:
        pw.run()
    except KeyboardInterrupt:
        print("\nPipeline stopped by user")
    except Exception as e:
        print(f"Pipeline error: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists(streaming_csv):
            os.remove(streaming_csv)
        print("Cleanup complete")

if __name__ == "__main__":
    main()
