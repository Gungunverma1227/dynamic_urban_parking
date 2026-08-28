import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from utils.haversine import compute_nearby_lots
from models.model1_baseline import baseline_linear_model
from models.model2_demand import demand_based_model
from models.model3_competitive import competitive_model
from utils.verify import verify_pricing
from utils.preprocess import load_and_preprocess

df = load_and_preprocess('dataset.csv')

# Visualize pricing over time for a sample lot
sample_lot = df['SystemCodeNumber'].iloc[np.random.randint(len(df['SystemCodeNumber']))]

df = baseline_linear_model(df, alpha=1.1, threshold_occupancy=0.6)

df_sample = df[df['SystemCodeNumber'] == sample_lot]

plt.figure(figsize=(10, 4))
plt.plot(df_sample['Timestamp'], df_sample['Price'], label='Price', color='darkorange')
plt.title(f'Baseline Pricing Over Time for Lot {sample_lot}')
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.grid(True)
plt.xticks(rotation=45)
plt.xlim([df['Timestamp'].iloc[0], df['Timestamp'].iloc[53]])
plt.tight_layout()
plt.show()

df_model2 = demand_based_model(df, lambda_scale=0.8, alpha=3.0)

df_sample = df_model2[df_model2['SystemCodeNumber'] == sample_lot]

fig, ax1 = plt.subplots(figsize=(12, 4))
ax1.plot(df_sample['Timestamp'], df_sample['Price'], color='purple')
ax1.set_ylabel('Price ($)', color='purple')

ax1.set_xlim([df['Timestamp'].iloc[0], df['Timestamp'].iloc[53]])

ax2 = ax1.twinx()
ax2.plot(df_sample['Timestamp'], df_sample['SmoothedDemand'], color='steelblue')
ax2.set_ylabel('Smoothed Demand', color='steelblue')

plt.title(f'Demand-Based Pricing vs Demand for Lot {sample_lot}')
fig.tight_layout()
plt.show()

# Extract unique lat/lon for each lot
df_meta = df_model2[['SystemCodeNumber', 'Latitude', 'Longitude']].drop_duplicates()

# Nearby lot mapping (within 1 km radius)
nearby_map = compute_nearby_lots(df_meta, radius_km=1.0)

df_model3 = competitive_model(df_model2, nearby_map, mu=0.05, lambda_scale=0.2)

df_sample = df_model3[df_model3['SystemCodeNumber'] == sample_lot]

# Plot Model 2 vs Model 3 price
plt.figure(figsize=(12, 4))
plt.plot(df_sample['Timestamp'], df_sample['Price'], label='Model 2: Demand-Based', color='orange')
plt.plot(df_sample['Timestamp'], df_sample['AdjustedPrice'], label='Model 3: Competitive Adjusted', color='purple')
plt.title(f'Pricing Comparison for Lot {sample_lot}')
plt.xlabel('Time')
plt.ylabel('Price ($)')
plt.legend()
plt.xticks(rotation=45)
# plt.xlim([df['Timestamp'].iloc[0], df['Timestamp'].iloc[53]])
plt.grid(True)
plt.tight_layout()
plt.show()

verify_pricing(df_model3, nearby_map=nearby_map)
