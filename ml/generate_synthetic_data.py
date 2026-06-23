"""
PULSE — Synthetic Data Generator
Generates realistic ASTrAM-like incident data for development and demo.
Based on exact distributions from the real dataset specification.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path

np.random.seed(42)

OUTPUT_PATH = Path(__file__).parent.parent / "data" / "Astram_event_data_anonymized.csv"


def generate_astram_data(n_incidents=8173):
    """Generate synthetic ASTrAM data matching real distribution."""

    # Date range: Nov 2023 to Apr 2024 (6 months)
    start_date = datetime(2023, 11, 1)
    end_date = datetime(2024, 4, 30)
    total_days = (end_date - start_date).days

    # Event type distribution: planned=467, unplanned=7706
    n_planned = 467
    n_unplanned = n_incidents - n_planned

    # Corridors with exact counts from spec
    corridors_dist = {
        'Mysore Road': 743,
        'Bellary Road 1': 610,
        'Tumkur Road': 458,
        'Bellary Road 2': 379,
        'Hosur Road': 298,
        'ORR North 1': 275,
        'Old Madras Road': 263,
        'Magadi Road': 245,
        'ORR East 1': 244,
    }
    # Remaining incidents distributed across other corridors
    remaining = n_incidents - sum(corridors_dist.values())
    per_other = remaining // 5
    other_corridors = {
        'Kanakapura Road': per_other,
        'Sarjapur Road': per_other,
        'Whitefield Road': per_other,
        'Bannerghatta Road': per_other,
        'NICE Road': remaining - 4 * per_other,
    }
    corridors_dist.update(other_corridors)

    # Event causes with counts
    causes_dist = {
        'vehicle_breakdown': 4896,
        'construction': 480,
        'water_logging': 458,
        'accident': 365,
        'tree_fall': 284,
        'public_event': 84,
        'procession': 72,
        'vip_movement': 20,
        'protest': 15,
    }

    # Junctions per corridor
    corridor_junctions = {
        'Mysore Road': ['Peenya Junction', 'Kengeri Flyover', 'Nayandahalli', 'Mysore Road Satellite Bus Stand', 'NICE Junction'],
        'Bellary Road 1': ['Mekhri Circle', 'Sadashivanagar', 'Hebbal Flyover', 'Palace Grounds', 'Sankey Tank Junction'],
        'Tumkur Road': ['Jalahalli Cross', 'Goraguntepalya', 'Yeshwanthpur', 'Peenya 2nd Stage', 'Dasarahalli'],
        'Bellary Road 2': ['Hebbal', 'Esteem Mall Junction', 'Yelahanka', 'Jakkur Cross', 'Sahakar Nagar'],
        'Hosur Road': ['Silk Board', 'Madiwala', 'Electronic City', 'Bommanahalli', 'BTM Layout'],
        'ORR North 1': ['Hebbal ORR', 'Nagawara', 'KR Puram', 'Thanisandra', 'Hennur'],
        'Old Madras Road': ['Indiranagar', 'Baiyyappanahalli', 'KR Puram', 'Marathahalli Bridge', 'Hoskote Junction'],
        'Magadi Road': ['Chord Road Junction', 'Kamakshipalya', 'Vijayanagar', 'Nayandahalli Circle', 'Magadi Road Station'],
        'ORR East 1': ['Marathahalli', 'Bellandur', 'Sarjapur Junction', 'Iblur', 'Agara'],
        'Kanakapura Road': ['Jayanagar', 'JP Nagar', 'Banashankari', 'Art of Living', 'Kanakapura Gate'],
        'Sarjapur Road': ['Wipro Gate', 'Bellandur Gate', 'Sarjapur Circle', 'Dommasandra', 'Carmelaram'],
        'Whitefield Road': ['Whitefield Station', 'ITPL', 'Kadugodi', 'Hoodi Circle', 'Varthur'],
        'Bannerghatta Road': ['Jayanagar 4th Block', 'Arekere', 'Gottigere', 'Hulimavu', 'Meenakshi Temple'],
        'NICE Road': ['NICE Tollgate', 'Kengeri Link', 'Mysore Road Exit', 'Tumkur Road Exit', 'Hosur Road Exit'],
    }

    # Zones
    zones = ['North', 'South', 'East', 'West', 'Central', 'Whitefield', 'Electronic City']
    corridor_zones = {
        'Mysore Road': 'West', 'Bellary Road 1': 'North', 'Tumkur Road': 'North',
        'Bellary Road 2': 'North', 'Hosur Road': 'South', 'ORR North 1': 'North',
        'Old Madras Road': 'East', 'Magadi Road': 'West', 'ORR East 1': 'East',
        'Kanakapura Road': 'South', 'Sarjapur Road': 'East', 'Whitefield Road': 'Whitefield',
        'Bannerghatta Road': 'South', 'NICE Road': 'West',
    }

    # Vehicle types for breakdowns
    veh_types = ['bmtc_bus', 'heavy_vehicle', 'lcv', 'truck']
    veh_weights = [0.3, 0.35, 0.15, 0.2]

    # Generate corridors based on distribution
    corridor_pool = []
    for corridor, count in corridors_dist.items():
        corridor_pool.extend([corridor] * count)
    np.random.shuffle(corridor_pool)
    corridor_pool = corridor_pool[:n_incidents]

    # Generate event causes based on distribution
    cause_pool = []
    for cause, count in causes_dist.items():
        cause_pool.extend([cause] * count)
    # Pad remaining
    while len(cause_pool) < n_incidents:
        cause_pool.append(np.random.choice(list(causes_dist.keys()), p=[v/sum(causes_dist.values()) for v in causes_dist.values()]))
    np.random.shuffle(cause_pool)
    cause_pool = cause_pool[:n_incidents]

    records = []

    for i in range(n_incidents):
        corridor = corridor_pool[i]
        cause = cause_pool[i]

        # Determine event type
        if cause in ['construction', 'public_event', 'procession', 'vip_movement', 'protest']:
            event_type = 'planned' if np.random.random() < 0.6 else 'unplanned'
        else:
            event_type = 'unplanned'

        # Generate time - breakdown peaks at 4-6 AM, construction spread during day
        if cause == 'vehicle_breakdown':
            # 59% of 4-6AM incidents are breakdowns - weight early hours
            hour = np.random.choice(
                range(24),
                p=_hour_weights_breakdown()
            )
        elif cause == 'construction':
            hour = np.random.choice(range(24), p=_hour_weights_construction())
        elif cause == 'water_logging':
            # Monsoon hours - afternoon/evening
            hour = np.random.choice(range(24), p=_hour_weights_waterlog())
        else:
            hour = np.random.randint(0, 24)

        # Random day within range
        day_offset = int(np.random.randint(0, total_days))
        start_dt = start_date + timedelta(days=day_offset, hours=int(hour), minutes=int(np.random.randint(0, 60)))

        # Duration based on cause
        if cause == 'construction':
            duration = max(30, np.random.normal(140, 60))  # median 140 min
        elif cause == 'vehicle_breakdown':
            duration = max(10, np.random.normal(41, 20))  # median 41 min
        elif cause == 'accident':
            duration = max(20, np.random.normal(75, 30))
        elif cause == 'water_logging':
            duration = max(30, np.random.normal(120, 50))
        else:
            duration = max(15, np.random.normal(60, 25))

        resolved_dt = start_dt + timedelta(minutes=duration)
        closed_dt = resolved_dt + timedelta(minutes=np.random.randint(5, 30))

        # Priority
        if cause in ['accident', 'construction'] or duration > 100:
            priority = 'High' if np.random.random() < 0.6 else 'Low'
        else:
            priority = 'High' if np.random.random() < 0.25 else 'Low'

        # Road closure - base 4.8%, higher with construction/accident
        closure_prob = 0.048
        if cause == 'construction':
            closure_prob = 0.15
        elif cause == 'accident':
            closure_prob = 0.12
        requires_closure = np.random.random() < closure_prob

        # Junction
        junctions = corridor_junctions.get(corridor, ['Unknown Junction'])
        junction = np.random.choice(junctions)

        # Zone
        zone = corridor_zones.get(corridor, 'Central')

        # GPS coords (approximate Bengaluru area)
        lat = 12.9716 + np.random.normal(0, 0.05)
        lon = 77.5946 + np.random.normal(0, 0.05)

        # Vehicle info for breakdowns
        veh_type = None
        age_of_truck = None
        reason_breakdown = None
        if cause == 'vehicle_breakdown':
            veh_type = np.random.choice(veh_types, p=veh_weights)
            if np.random.random() < 0.05:  # 276 out of ~4900
                age_of_truck = np.random.randint(5, 25)
                reason_breakdown = np.random.choice([
                    'engine_failure', 'tyre_burst', 'brake_failure',
                    'overheating', 'fuel_issue', 'electrical_failure'
                ])

        records.append({
            'event_type': event_type,
            'event_cause': cause,
            'start_datetime': start_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'closed_datetime': closed_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'resolved_datetime': resolved_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'corridor': corridor,
            'priority': priority,
            'requires_road_closure': requires_closure,
            'junction': junction,
            'zone': zone,
            'latitude': round(lat, 6),
            'longitude': round(lon, 6),
            'veh_type': veh_type,
            'age_of_truck': age_of_truck,
            'reason_breakdown': reason_breakdown,
        })

    df = pd.DataFrame(records)

    # Ensure planned count is close to 467
    planned_count = (df['event_type'] == 'planned').sum()
    if planned_count < 467:
        # Convert some unplanned construction/events to planned
        unplanned_convertible = df[
            (df['event_type'] == 'unplanned') &
            (df['event_cause'].isin(['construction', 'public_event', 'procession', 'vip_movement']))
        ].index
        convert_n = min(467 - planned_count, len(unplanned_convertible))
        df.loc[unplanned_convertible[:convert_n], 'event_type'] = 'planned'

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated {len(df)} incidents → {OUTPUT_PATH}")
    print(f"  Planned: {(df['event_type']=='planned').sum()}")
    print(f"  Unplanned: {(df['event_type']=='unplanned').sum()}")
    print(f"  Corridors: {df['corridor'].nunique()}")
    print(f"  Date range: {df['start_datetime'].min()} to {df['start_datetime'].max()}")

    return df


def _hour_weights_breakdown():
    """Hour distribution for breakdowns - peaks at 4-6 AM."""
    w = np.ones(24) * 2
    w[4:7] = 10  # early surge - 59% of 4-6AM incidents are breakdowns
    w[7:10] = 5  # morning commute
    w[17:21] = 4  # evening
    w[0:4] = 3   # late night
    return w / w.sum()


def _hour_weights_construction():
    """Construction mostly during working hours."""
    w = np.ones(24) * 1
    w[8:18] = 5
    w[22:24] = 3  # night work
    w[0:5] = 3
    return w / w.sum()


def _hour_weights_waterlog():
    """Water logging peaks in afternoon/evening (monsoon pattern)."""
    w = np.ones(24) * 1
    w[14:20] = 6
    w[20:23] = 3
    return w / w.sum()


if __name__ == "__main__":
    print("=" * 60)
    print("PULSE — Generating Synthetic ASTrAM Data")
    print("=" * 60)
    df = generate_astram_data()

    # Validation stats
    print(f"\n--- Validation ---")
    print(f"Event causes:")
    print(df['event_cause'].value_counts().to_string())
    print(f"\nTop corridors:")
    print(df['corridor'].value_counts().head(9).to_string())
    print(f"\nPriority: {df['priority'].value_counts().to_dict()}")
    print(f"Road closure rate: {df['requires_road_closure'].mean()*100:.1f}%")
