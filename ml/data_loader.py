"""
PULSE — Data Loader and Preprocessor
Loads ASTrAM event data and computes baseline corridor risk matrices.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json


DATA_PATH = Path(__file__).parent.parent / "data" / "Astram_event_data_anonymized.csv"


def load_astram_data(path=None):
    """Load and clean ASTrAM dataset."""
    if path is None:
        path = DATA_PATH

    df = pd.read_csv(path)

    # Parse datetime columns
    for col in ['start_datetime', 'closed_datetime', 'resolved_datetime', 'end_datetime', 'created_date', 'modified_datetime']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)

    # Standardize event_type
    df['event_type'] = df['event_type'].str.lower().str.strip()

    # Standardize event_cause
    df['event_cause'] = df['event_cause'].str.lower().str.strip().str.replace(' ', '_').str.replace('/', '_')

    # Compute duration in minutes — use resolved_datetime first, fallback to closed_datetime
    df['duration_minutes'] = np.nan
    if 'resolved_datetime' in df.columns and 'start_datetime' in df.columns:
        df['duration_minutes'] = (
            df['resolved_datetime'] - df['start_datetime']
        ).dt.total_seconds() / 60.0

    # Where resolved is missing, use closed_datetime
    mask = df['duration_minutes'].isna() & df['closed_datetime'].notna()
    df.loc[mask, 'duration_minutes'] = (
        df.loc[mask, 'closed_datetime'] - df.loc[mask, 'start_datetime']
    ).dt.total_seconds() / 60.0

    # Remove negative or extreme durations
    df.loc[df['duration_minutes'] < 0, 'duration_minutes'] = np.nan
    df.loc[df['duration_minutes'] > 1440, 'duration_minutes'] = np.nan  # cap at 24h

    # Extract temporal features
    df['hour'] = df['start_datetime'].dt.hour
    df['day_of_week'] = df['start_datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['date'] = df['start_datetime'].dt.date

    # Time blocks
    def get_time_block(hour):
        if 0 <= hour < 4:
            return 0  # night
        elif 4 <= hour < 7:
            return 1  # early_surge
        elif 7 <= hour < 10:
            return 2  # morning
        elif 10 <= hour < 17:
            return 3  # day
        elif 17 <= hour < 22:
            return 4  # evening_surge
        else:
            return 5  # late_night

    df['time_block'] = df['hour'].apply(get_time_block)

    # Priority encoding
    df['is_high_priority'] = (df['priority'].str.lower() == 'high').astype(int)

    # Filter to named corridors only (exclude 'Non-corridor') for corridor-based analysis
    # Keep all rows but mark corridor ones
    df['is_corridor'] = (~df['corridor'].isin(['Non-corridor', 'non-corridor'])) & df['corridor'].notna()

    # Road closure encoding
    df['requires_closure'] = df['requires_road_closure'].fillna(False)
    if df['requires_closure'].dtype == object:
        df['requires_closure'] = df['requires_closure'].astype(str).str.upper().map(
            {'TRUE': 1, 'FALSE': 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0,
             '1': 1, '0': 0, '1.0': 1, '0.0': 0, 'NAN': 0, 'NONE': 0}
        ).fillna(0).astype(int)
    else:
        df['requires_closure'] = df['requires_closure'].astype(int)

    return df


def compute_corridor_stats(df):
    """Compute baseline corridor statistics."""
    stats = {}

    corridors = df['corridor'].dropna().unique()

    for corridor in corridors:
        cdf = df[df['corridor'] == corridor]
        total = len(cdf)
        hours_span = max(1, (cdf['start_datetime'].max() - cdf['start_datetime'].min()).total_seconds() / 3600)

        stats[corridor] = {
            'total_incidents': total,
            'base_rate_per_hour': total / hours_span,
            'breakdown_pct': len(cdf[cdf['event_cause'] == 'vehicle_breakdown']) / max(1, total),
            'construction_pct': len(cdf[cdf['event_cause'] == 'construction']) / max(1, total),
            'high_priority_pct': cdf['is_high_priority'].mean(),
            'median_duration': cdf['duration_minutes'].median(),
            'closure_rate': cdf['requires_closure'].mean(),
            'peak_hours': cdf.groupby('hour').size().nlargest(3).index.tolist(),
            'hourly_rates': cdf.groupby('hour').size().reindex(range(24), fill_value=0).to_dict(),
        }

    return stats


def compute_cascade_multiplier(df):
    """
    Compute cascade multiplier: unplanned incident rate after planned events
    vs baseline unplanned rate on same corridor.
    """
    planned = df[df['event_type'] == 'planned'].copy()
    unplanned = df[df['event_type'] == 'unplanned'].copy()

    cascade_windows = []

    for _, event in planned.iterrows():
        corridor = event['corridor']
        start = event['start_datetime']
        window_end = start + pd.Timedelta(hours=4)

        # Count unplanned incidents on same corridor in 4h window after planned event start
        mask = (
            (unplanned['corridor'] == corridor) &
            (unplanned['start_datetime'] >= start) &
            (unplanned['start_datetime'] <= window_end)
        )
        count = mask.sum()
        cascade_windows.append({
            'corridor': corridor,
            'planned_start': start,
            'unplanned_count_4h': count,
            'event_cause': event['event_cause']
        })

    cascade_df = pd.DataFrame(cascade_windows)

    # Baseline: average unplanned incidents per 4h window per corridor (without planned events)
    total_hours = (df['start_datetime'].max() - df['start_datetime'].min()).total_seconds() / 3600
    corridors = df['corridor'].dropna().unique()

    baseline_rates = {}
    for corridor in corridors:
        corridor_unplanned = len(unplanned[unplanned['corridor'] == corridor])
        windows = total_hours / 4
        baseline_rates[corridor] = corridor_unplanned / max(1, windows)

    # Average post-planned rate
    post_planned_rate = cascade_df['unplanned_count_4h'].mean()
    overall_baseline = np.mean(list(baseline_rates.values()))

    multiplier = post_planned_rate / max(0.001, overall_baseline)

    return {
        'cascade_multiplier': round(multiplier, 2),
        'post_planned_rate': round(post_planned_rate, 3),
        'baseline_rate': round(overall_baseline, 3),
        'total_planned_events_analyzed': len(cascade_df),
        'corridor_baselines': {k: round(v, 3) for k, v in baseline_rates.items()},
    }


def find_case_study(df):
    """
    Find a real incident where preconditions (construction + high base rate + patterns)
    were visible hours before a high-priority incident occurred.
    """
    # Look for high-priority unplanned incidents on corridors with active construction
    construction = df[df['event_cause'] == 'construction'].copy()
    high_priority_unplanned = df[
        (df['event_type'] == 'unplanned') &
        (df['is_high_priority'] == 1) &
        (df['duration_minutes'].notna())  # prefer ones with duration data
    ].copy()

    candidates = []

    for _, incident in high_priority_unplanned.iterrows():
        corridor = incident['corridor']
        incident_time = incident['start_datetime']

        if pd.isna(incident_time) or pd.isna(corridor):
            continue

        # Check if construction was active on this corridor at the time
        active_construction = construction[
            (construction['corridor'] == corridor) &
            (construction['start_datetime'] < incident_time) &
            (
                (construction['resolved_datetime'] > incident_time) |
                (construction['closed_datetime'] > incident_time) |
                (construction['resolved_datetime'].isna() & construction['closed_datetime'].isna())
            )
        ]

        if len(active_construction) > 0:
            # How many historical breakdowns in this hour/corridor combo?
            hist_breakdowns = len(df[
                (df['corridor'] == corridor) &
                (df['hour'] == incident['hour']) &
                (df['event_cause'] == 'vehicle_breakdown') &
                (df['start_datetime'] < incident_time)
            ])

            construction_event = active_construction.iloc[0]
            construction_days = (incident_time - construction_event['start_datetime']).days

            junction_val = incident.get('junction', None)
            if pd.isna(junction_val):
                junction_val = None

            candidates.append({
                'incident_date': incident_time.strftime('%Y-%m-%d'),
                'incident_time': incident_time.strftime('%H:%M'),
                'corridor': corridor,
                'event_cause': incident['event_cause'],
                'duration_minutes': float(incident['duration_minutes']) if pd.notna(incident['duration_minutes']) else None,
                'requires_closure': bool(incident['requires_closure']),
                'construction_active': True,
                'construction_days': int(construction_days),
                'historical_breakdowns_same_hour': int(hist_breakdowns),
                'hour': int(incident['hour']),
                'junction': junction_val,
            })

    # Sort by most compelling (most preconditions visible, prefer those with junctions)
    candidates.sort(key=lambda x: (
        x['junction'] is not None,
        x['historical_breakdowns_same_hour'],
        x['duration_minutes'] or 0,
    ), reverse=True)

    return candidates[0] if candidates else None


if __name__ == "__main__":
    print("Loading ASTrAM data...")
    df = load_astram_data()
    print(f"Loaded {len(df)} incidents")
    print(f"Date range: {df['start_datetime'].min()} to {df['start_datetime'].max()}")
    print(f"\nEvent types: {df['event_type'].value_counts().to_dict()}")
    print(f"Event causes: {df['event_cause'].value_counts().head(10).to_dict()}")

    print("\nComputing corridor stats...")
    stats = compute_corridor_stats(df)
    print(f"Corridors analyzed: {len(stats)}")

    print("\nComputing cascade multiplier...")
    cascade = compute_cascade_multiplier(df)
    print(f"Cascade multiplier: {cascade['cascade_multiplier']}x")
    print(f"Post-planned rate: {cascade['post_planned_rate']} per 4h window")
    print(f"Baseline rate: {cascade['baseline_rate']} per 4h window")

    print("\nSearching for case study...")
    case = find_case_study(df)
    if case:
        print(f"Found: {case['corridor']} on {case['incident_date']}")
        print(f"  Construction active for {case['construction_days']} days")
        print(f"  Historical breakdowns same hour: {case['historical_breakdowns_same_hour']}")
    else:
        print("No ideal case study found - will generate from patterns")
