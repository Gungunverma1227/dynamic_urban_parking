import pandas as pd
import numpy as np

def baseline_linear_model(
        df, 
        alpha=2.0, threshold_occupancy=0.5, 
        base_price=10.0, 
        min_price=5.0, max_price=20.0, 
):
    df = df.sort_values(by=['SystemCodeNumber', 'Timestamp']).copy()

    df['Price'] = np.nan

    for lot in df['SystemCodeNumber'].unique():
        lot_df = df[df['SystemCodeNumber'] == lot].copy()
        prices = [base_price]

        for i in range(1, len(lot_df)):
            prev_price = prices[-1]
            occupancy_rate = lot_df.iloc[i]['OccupancyRate']
            new_price = prev_price + alpha * (occupancy_rate - threshold_occupancy)
            new_price = max(min(new_price, max_price), min_price)
            prices.append(new_price)

        df.loc[df['SystemCodeNumber'] == lot, 'Price'] = prices

    return df
