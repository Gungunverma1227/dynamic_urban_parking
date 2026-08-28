import pandas as pd
import numpy as np

def competitive_model(
    df, 
    nearby_map, 
    mu=0.3, 
    base_price=10, 
    smoothing_factor=0.1, 
    lambda_scale=0.8, 
    min_price=5, max_price=20
):
    df = df.copy()

    adjusted_demand = []

    for t in df['Timestamp'].unique():
        snapshot = df[df['Timestamp'] == t].set_index('SystemCodeNumber')

        for lot in snapshot.index:
            demand = snapshot.loc[lot, 'SmoothedDemand']
            own_price = snapshot.loc[lot, 'Price']
            neighbors = nearby_map.get(lot, [])
            neighbors = [n for n in neighbors if n in snapshot.index]

            if neighbors:
                comp_prices = snapshot.loc[neighbors, 'Price'].values
                comp_avg = comp_prices.mean()
                demand -= mu * np.tanh(own_price - comp_avg)

            adjusted_demand.append({
                'SystemCodeNumber': lot,
                'Timestamp': t,
                'AdjustedDemand': demand
            })

    df_adj = pd.DataFrame(adjusted_demand)
    df_final = df.merge(df_adj, on=['SystemCodeNumber', 'Timestamp'])
    df_final['AdjustedDemandNorm'] = df_final.groupby('SystemCodeNumber')['AdjustedDemand'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-5) * 2 - 1
    )
    df_final = df_final.sort_values(by=['SystemCodeNumber', 'Timestamp'])
    df_final['AdjustedPrice'] = np.nan

    for lot in df_final['SystemCodeNumber'].unique():
        lot_df = df_final[df_final['SystemCodeNumber'] == lot].copy()
        prices = [base_price]

        for i in range(1, len(lot_df)):
            prev_price = prices[-1]
            delta_price = lambda_scale * lot_df.iloc[i]['AdjustedDemandNorm']
            new_price = prev_price*(1-smoothing_factor) + base_price*(1+delta_price)*smoothing_factor

            new_price = min(max(new_price, min_price), max_price)
            prices.append(new_price)

        df_final.loc[df_final['SystemCodeNumber'] == lot, 'AdjustedPrice'] = prices
    
    return df_final
