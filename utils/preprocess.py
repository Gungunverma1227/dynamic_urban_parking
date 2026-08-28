import pandas as pd
import numpy as np

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    df['Timestamp'] = pd.to_datetime(df['LastUpdatedDate'] + ' ' + df['LastUpdatedTime'], format="%d-%m-%Y %H:%M:%S")

    df = df.sort_values(by=['SystemCodeNumber', 'Timestamp'])

    df.drop(['LastUpdatedDate', 'LastUpdatedTime'], axis=1, inplace=True)

    df['OccupancyRate'] = df['Occupancy'] / df['Capacity']

    vehicle_type_map = {'car': 0.6, 'bike': 0.2, 'truck': 1}
    df['VehicleTypeWeight'] = df['VehicleType'].map(vehicle_type_map)

    traffic_map = {'low': 0, 'average': 0.5, 'high': 1}
    df['TrafficLevel'] = df['TrafficConditionNearby'].map(traffic_map)

    df.drop(['VehicleType', 'TrafficConditionNearby'], axis=1, inplace=True)

    df['QueueLengthNorm'] = df['QueueLength'] / (df['QueueLength'].max() + 1e-5)

    df = df.groupby(['SystemCodeNumber', 'Timestamp'], as_index=False).mean()

    df.ffill(inplace=True)

    processed_df = df[[
        'SystemCodeNumber',
        'Timestamp',
        'Latitude',
        'Longitude',
        'Capacity',
        'Occupancy',
        'OccupancyRate',
        'QueueLength',
        'QueueLengthNorm',
        'VehicleTypeWeight',
        'TrafficLevel',
        'IsSpecialDay'
    ]].copy()

    return processed_df
