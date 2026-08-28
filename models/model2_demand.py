import pandas as pd
import numpy as np

def demand_based_model(
    df,
    base_price=10.0,
    alpha=0.5, beta=0.2, gamma=0.15, delta=1.0, epsilon=0.5,
    lambda_scale=0.8,
    min_price=5,
    max_price=20,
    smoothing_factor=0.1
):
    df = df.sort_values(by=['SystemCodeNumber', 'Timestamp']).copy()

    features = ['OccupancyRate', 'QueueLength', 'TrafficLevel', 'VehicleTypeWeight']
    for feat in features:
        df[f'{feat}_s'] = df.groupby('SystemCodeNumber')[feat].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )

    df['RawDemand'] = (
        alpha * df['OccupancyRate']
        + beta * df['QueueLength']
        + gamma * df['TrafficLevel']
        + delta * df['IsSpecialDay']
        + epsilon * df['VehicleTypeWeight']
    )

    df['NormalizedDemand'] = df.groupby('SystemCodeNumber')['RawDemand'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-5) * 2 - 1
    )
    df['SmoothedDemand'] = df.groupby('SystemCodeNumber')['NormalizedDemand'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )

    df['Price'] = np.nan

    for lot in df['SystemCodeNumber'].unique():
        lot_df = df[df['SystemCodeNumber'] == lot].copy()
        prices = [base_price]

        for i in range(1, len(lot_df)):
            prev_price = prices[-1]
            delta_price = lambda_scale * lot_df.iloc[i]['SmoothedDemand']
            new_price = prev_price*(1-smoothing_factor) + base_price*(1+delta_price)*smoothing_factor

            new_price = min(max(new_price, min_price), max_price)
            prices.append(new_price)

        df.loc[df['SystemCodeNumber'] == lot, 'Price'] = prices

    return df
