import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def compute_nearby_lots(df_meta, radius_km=1.0):
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
