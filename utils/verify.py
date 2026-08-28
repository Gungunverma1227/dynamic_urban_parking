import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def verify_pricing(df, nearby_map=None, min_price=5, max_price=20, base_price=10):
    # 1. Range check
    print("1. Price Range Stats:")
    print(df['Price'].describe())
    assert df['Price'].between(min_price, max_price).all(), "Prices exceed defined bounds!"

    pct_min = (df['Price'] <= min_price).mean() * 100
    pct_max = (df['Price'] >= max_price).mean() * 100
    print(f"{pct_min:.1f}% of prices hit the minimum (${min_price})")
    print(f"{pct_max:.1f}% of prices hit the maximum (${max_price})")

    # 2. Smoothness (price delta)
    print("\n2. Price Volatility:")
    df['PriceDelta'] = df.groupby('SystemCodeNumber')['Price'].diff()
    delta_stats = df['PriceDelta'].abs().describe()
    print(delta_stats)

    # 3. Correlation with occupancy
    print("\n3. Occupancy ↔ Price Correlation:")
    corr_list = []
    for lot in df['SystemCodeNumber'].unique():
        sub = df[df['SystemCodeNumber'] == lot]
        if sub['OccupancyRate'].nunique() > 1:
            corr = sub['OccupancyRate'].corr(sub['Price'])
            corr_list.append((lot, corr))
    corr_series = pd.Series(dict(corr_list))
    print(corr_series.describe())
    print("Mean correlation (should be positive):", corr_series.mean())

    # 4. Competitive consistency
    if nearby_map is not None:
        print("\n4. Competitive Consistency (Sample):")
        sample_ts = df['Timestamp'].unique()[::max(1, len(df['Timestamp'].unique()) // 10)]
        for t in sample_ts:
            snap = df[df['Timestamp'] == t].set_index('SystemCodeNumber')
            for lot in snap.index:
                own = snap.at[lot, 'Price']
                neighbors = nearby_map.get(lot, [])
                neighbors = [n for n in neighbors if n in snap.index]
                if not neighbors:
                    continue
                avg_neighbor = snap.loc[neighbors]['Price'].mean()
                diff = own - avg_neighbor
                print(f"{t} | {lot}: Price = ${own:.2f}, Market Avg = ${avg_neighbor:.2f}, Δ = {diff:+.2f}")

    # 5. Visualize one random lot
    sample_lot = np.random.choice(df['SystemCodeNumber'].unique())
    print(f"\n5. Plotting sample lot: {sample_lot}")
    df_sample = df[df['SystemCodeNumber'] == sample_lot]

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(df_sample['Timestamp'], df_sample['Price'], color='purple', label='Price')
    ax1.set_ylabel('Price', color='purple')
    ax1.tick_params(axis='y', labelcolor='purple')

    ax2 = ax1.twinx()
    ax2.plot(df_sample['Timestamp'], df_sample['OccupancyRate'], color='steelblue', label='Occupancy')
    ax2.set_ylabel('Occupancy Rate', color='steelblue')
    ax2.tick_params(axis='y', labelcolor='steelblue')

    plt.title(f'Lot {sample_lot} — Price vs Occupancy')
    fig.tight_layout()
    plt.show()

    print("\nVerification complete.")
