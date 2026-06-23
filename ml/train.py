"""
PULSE — Model Training Pipeline
GradientBoostingClassifier for corridor risk prediction.
Temporal train/validation split: Nov-Feb train, Mar-Apr validate.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, roc_auc_score
)
import pickle
import json
from pathlib import Path
from data_loader import load_astram_data, compute_corridor_stats, compute_cascade_multiplier

MODEL_DIR = Path(__file__).parent.parent / "backend" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def engineer_features(df, corridor_stats, cascade_data):
    """Engineer all features for the prediction model."""

    le_corridor = LabelEncoder()
    df['corridor_encoded'] = le_corridor.fit_transform(df['corridor'].fillna('unknown'))

    # Corridor base features
    df['corridor_base_rate'] = df['corridor'].map(
        lambda c: corridor_stats.get(c, {}).get('base_rate_per_hour', 0)
    )
    df['corridor_breakdown_pct'] = df['corridor'].map(
        lambda c: corridor_stats.get(c, {}).get('breakdown_pct', 0)
    )
    df['corridor_construction_pct'] = df['corridor'].map(
        lambda c: corridor_stats.get(c, {}).get('construction_pct', 0)
    )

    # Cascade features — THE KEY DIFFERENTIATOR
    # For each row, count active planned events on same corridor within past 4h
    df = df.sort_values('start_datetime').reset_index(drop=True)

    planned_mask = df['event_type'] == 'planned'
    planned_events = df[planned_mask].copy()

    # Vectorized cascade feature computation
    active_planned = np.zeros(len(df))
    hours_since_planned = np.full(len(df), 999.0)
    concurrent_incidents = np.zeros(len(df))

    for idx, row in df.iterrows():
        corridor = row['corridor']
        t = row['start_datetime']

        if pd.isna(t) or pd.isna(corridor):
            continue

        # Active planned events on same corridor in past 4 hours
        mask = (
            planned_events['corridor'] == corridor
        ) & (
            planned_events['start_datetime'] <= t
        ) & (
            planned_events['start_datetime'] >= t - pd.Timedelta(hours=4)
        )
        active_planned[idx] = mask.sum()

        # Hours since last planned event on corridor
        past_planned = planned_events[
            (planned_events['corridor'] == corridor) &
            (planned_events['start_datetime'] <= t)
        ]
        if len(past_planned) > 0:
            last_planned_time = past_planned['start_datetime'].max()
            hours_since_planned[idx] = (t - last_planned_time).total_seconds() / 3600

        # Concurrent incidents on same corridor (within 1 hour)
        concurrent = (
            (df['corridor'] == corridor) &
            (df['start_datetime'] >= t - pd.Timedelta(hours=1)) &
            (df['start_datetime'] <= t) &
            (df.index != idx)
        )
        concurrent_incidents[idx] = concurrent.sum()

    df['active_planned_events'] = active_planned
    df['hours_since_planned'] = hours_since_planned.clip(0, 168)  # cap at 1 week
    df['concurrent_incidents'] = concurrent_incidents

    # Cascade risk score
    baseline_rate = df['corridor_base_rate']
    multiplier = cascade_data.get('cascade_multiplier', 1.81)
    df['cascade_risk_score'] = df['active_planned_events'] * baseline_rate * multiplier

    # Active construction feature
    construction_df = df[df['event_cause'] == 'construction'].copy()
    has_construction = np.zeros(len(df))

    for idx, row in df.iterrows():
        corridor = row['corridor']
        t = row['start_datetime']
        if pd.isna(t) or pd.isna(corridor):
            continue

        active = construction_df[
            (construction_df['corridor'] == corridor) &
            (construction_df['start_datetime'] <= t) &
            (
                (construction_df['resolved_datetime'] >= t) |
                (construction_df['closed_datetime'] >= t) |
                (construction_df['resolved_datetime'].isna() & construction_df['closed_datetime'].isna())
            )
        ]
        has_construction[idx] = 1 if len(active) > 0 else 0

    df['has_active_construction'] = has_construction

    return df, le_corridor


def create_target(df, lookahead_hours=4):
    """
    Target: will there be a high-priority incident on this corridor
    within the next lookahead_hours?
    """
    target = np.zeros(len(df))

    high_priority = df[df['is_high_priority'] == 1].copy()

    for idx, row in df.iterrows():
        corridor = row['corridor']
        t = row['start_datetime']
        if pd.isna(t) or pd.isna(corridor):
            continue

        future_high = high_priority[
            (high_priority['corridor'] == corridor) &
            (high_priority['start_datetime'] > t) &
            (high_priority['start_datetime'] <= t + pd.Timedelta(hours=lookahead_hours)) &
            (high_priority.index != idx)
        ]
        if len(future_high) > 0:
            target[idx] = 1

    df['target_high_risk_4h'] = target
    return df


def train_model(df, corridor_stats, cascade_data):
    """Train the cascade prediction model with temporal split."""

    print("Engineering features...")
    df, le_corridor = engineer_features(df, corridor_stats, cascade_data)

    print("Creating target variable...")
    df = create_target(df)

    # Feature columns
    feature_cols = [
        'hour', 'day_of_week', 'is_weekend', 'time_block',
        'corridor_encoded', 'corridor_base_rate', 'corridor_breakdown_pct',
        'corridor_construction_pct',
        'active_planned_events', 'hours_since_planned',
        'concurrent_incidents', 'cascade_risk_score',
        'has_active_construction',
    ]

    # Temporal split: train on Nov 2023 - Feb 2024, validate Mar-Apr 2024
    split_date = pd.Timestamp('2024-03-01', tz='UTC')
    train_mask = df['start_datetime'] < split_date
    val_mask = df['start_datetime'] >= split_date

    X_train = df.loc[train_mask, feature_cols].fillna(0)
    y_train = df.loc[train_mask, 'target_high_risk_4h']
    X_val = df.loc[val_mask, feature_cols].fillna(0)
    y_val = df.loc[val_mask, 'target_high_risk_4h']

    print(f"Train set: {len(X_train)} samples, {y_train.sum():.0f} positive ({y_train.mean()*100:.1f}%)")
    print(f"Val set: {len(X_val)} samples, {y_val.sum():.0f} positive ({y_val.mean()*100:.1f}%)")

    # Train GradientBoosting classifier
    model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_val, y_prob)
    except ValueError:
        auc = 0.0

    print(f"\n{'='*50}")
    print(f"MODEL PERFORMANCE (Temporal Validation)")
    print(f"{'='*50}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    print(f"AUC-ROC:   {auc:.3f}")
    print(f"\n{classification_report(y_val, y_pred, target_names=['Low Risk', 'High Risk'])}")

    # Feature importance
    importances = dict(zip(feature_cols, model.feature_importances_))
    importances_sorted = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    print("\nFeature Importances:")
    for feat, imp in importances_sorted.items():
        print(f"  {feat:30s} {imp:.4f}")

    # Train duration regression model
    duration_df = df[df['duration_minutes'].notna() & (df['duration_minutes'] > 0)].copy()
    X_dur = duration_df[feature_cols].fillna(0)
    y_dur = duration_df['duration_minutes'].clip(0, 480)  # cap at 8h

    duration_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    duration_model.fit(X_dur, y_dur)

    # Save everything
    artifacts = {
        'model': model,
        'duration_model': duration_model,
        'label_encoder': le_corridor,
        'feature_cols': feature_cols,
        'corridor_stats': corridor_stats,
        'cascade_data': cascade_data,
        'metrics': {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'auc': round(auc, 4),
            'train_size': int(len(X_train)),
            'val_size': int(len(X_val)),
            'positive_rate_train': round(float(y_train.mean()), 4),
            'positive_rate_val': round(float(y_val.mean()), 4),
        },
        'feature_importances': {k: round(float(v), 4) for k, v in importances_sorted.items()},
    }

    model_path = MODEL_DIR / "pulse_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"\nModel saved to {model_path}")

    # Save metrics as JSON for frontend
    metrics_path = MODEL_DIR / "model_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'metrics': artifacts['metrics'],
            'feature_importances': artifacts['feature_importances'],
            'corridor_stats': {k: {kk: (vv if not isinstance(vv, (np.floating, np.integer)) else float(vv))
                                    for kk, vv in v.items()}
                               for k, v in corridor_stats.items()},
            'cascade_data': cascade_data,
        }, f, indent=2, default=str)
    print(f"Metrics saved to {metrics_path}")

    return artifacts


if __name__ == "__main__":
    from data_loader import load_astram_data, compute_corridor_stats, compute_cascade_multiplier

    print("=" * 60)
    print("PULSE — Training Pipeline")
    print("=" * 60)

    df = load_astram_data()
    print(f"Loaded {len(df)} incidents")

    # Filter to named corridors only for model training
    df = df[df['is_corridor']].copy()
    print(f"Corridor incidents: {len(df)}")

    corridor_stats = compute_corridor_stats(df)
    cascade_data = compute_cascade_multiplier(df)
    print(f"Cascade multiplier: {cascade_data['cascade_multiplier']}x")

    artifacts = train_model(df, corridor_stats, cascade_data)
    print("\n✓ Training complete")
