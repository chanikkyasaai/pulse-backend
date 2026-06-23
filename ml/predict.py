"""
PULSE — Prediction Engine
Generates corridor risk scores for the next 24 hours.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime, timedelta

MODEL_DIR = Path(__file__).parent.parent / "models"


def load_model():
    """Load trained model artifacts."""
    model_path = MODEL_DIR / "pulse_model.pkl"
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def generate_risk_scores(artifacts, weather_data=None, planned_events=None, target_date=None):
    """
    Generate corridor risk scores for next 24 hours.

    Args:
        artifacts: trained model artifacts dict
        weather_data: dict with hourly precipitation, temperature, visibility
        planned_events: list of planned events with corridor, start_time, type
        target_date: date to predict for (defaults to tomorrow)

    Returns:
        List of corridor risk assessments
    """
    model = artifacts['model']
    duration_model = artifacts['duration_model']
    le_corridor = artifacts['label_encoder']
    feature_cols = artifacts['feature_cols']
    corridor_stats = artifacts['corridor_stats']
    cascade_data = artifacts['cascade_data']

    if target_date is None:
        target_date = datetime.now() + timedelta(days=1)

    corridors = list(corridor_stats.keys())
    predictions = []

    for corridor in corridors:
        stats = corridor_stats[corridor]

        # Generate predictions for each 4-hour block
        time_blocks = [
            (0, 4, 'night', 0),
            (4, 7, 'early_surge', 1),
            (7, 10, 'morning', 2),
            (10, 17, 'day', 3),
            (17, 22, 'evening_surge', 4),
            (22, 24, 'late_night', 5),
        ]

        corridor_risks = []

        for start_h, end_h, block_name, block_id in time_blocks:
            # Encode corridor
            try:
                corridor_enc = le_corridor.transform([corridor])[0]
            except ValueError:
                corridor_enc = 0

            # Check for active planned events on this corridor
            active_planned = 0
            hours_since_planned = 168.0  # default 1 week

            if planned_events:
                for event in planned_events:
                    if event.get('corridor') == corridor:
                        event_start = pd.to_datetime(event.get('start_time'), utc=True)
                        block_start = pd.Timestamp(target_date.replace(hour=start_h, minute=0), tz='UTC')
                        try:
                            if event_start <= block_start:
                                active_planned += 1
                                hrs = (block_start - event_start).total_seconds() / 3600
                                hours_since_planned = min(hours_since_planned, hrs)
                        except Exception:
                            pass

            # Check for active construction
            has_construction = 0
            if planned_events:
                for event in planned_events:
                    if (event.get('corridor') == corridor and
                        event.get('type', '').lower() == 'construction'):
                        has_construction = 1

            # Cascade risk score
            multiplier = cascade_data.get('cascade_multiplier', 1.81)
            cascade_risk = active_planned * stats['base_rate_per_hour'] * multiplier

            # Build feature vector
            mid_hour = (start_h + end_h) // 2
            dow = target_date.weekday() if hasattr(target_date, 'weekday') else 0
            is_weekend = 1 if dow in [5, 6] else 0

            features = {
                'hour': mid_hour,
                'day_of_week': dow,
                'is_weekend': is_weekend,
                'time_block': block_id,
                'corridor_encoded': corridor_enc,
                'corridor_base_rate': stats['base_rate_per_hour'],
                'corridor_breakdown_pct': stats['breakdown_pct'],
                'corridor_construction_pct': stats['construction_pct'],
                'active_planned_events': active_planned,
                'hours_since_planned': hours_since_planned,
                'concurrent_incidents': 0,  # no live data for prediction
                'cascade_risk_score': cascade_risk,
                'has_active_construction': has_construction,
            }

            X = pd.DataFrame([features])[feature_cols]

            # Predict probability and duration
            prob = model.predict_proba(X)[0][1]
            predicted_duration = duration_model.predict(X)[0]

            # Risk level classification
            if prob >= 0.75:
                risk_level = 'CRITICAL'
            elif prob >= 0.55:
                risk_level = 'HIGH'
            elif prob >= 0.35:
                risk_level = 'MODERATE'
            else:
                risk_level = 'LOW'

            # Determine drivers
            drivers = []
            if has_construction:
                drivers.append('Active construction on corridor')
            if active_planned > 0:
                drivers.append(f'{active_planned} planned event(s) triggering cascade')
            if stats['breakdown_pct'] > 0.6:
                drivers.append(f"High breakdown corridor ({stats['breakdown_pct']*100:.0f}%)")
            if mid_hour in stats.get('peak_hours', []):
                drivers.append('Peak incident hour historically')
            if weather_data:
                rain = weather_data.get('precipitation', {}).get(str(mid_hour), 0)
                if rain > 0.5:
                    drivers.append(f'Rain forecast: {rain:.1f}mm')

            corridor_risks.append({
                'time_block': block_name,
                'start_hour': start_h,
                'end_hour': end_h,
                'probability': round(prob, 3),
                'risk_level': risk_level,
                'predicted_duration_min': round(predicted_duration, 0),
                'drivers': drivers,
                'cascade_active': active_planned > 0,
            })

        # Overall corridor risk = max risk across all blocks
        max_risk = max(corridor_risks, key=lambda x: x['probability'])

        predictions.append({
            'corridor': corridor,
            'overall_risk_level': max_risk['risk_level'],
            'overall_probability': max_risk['probability'],
            'peak_risk_window': f"{max_risk['start_hour']:02d}:00-{max_risk['end_hour']:02d}:00",
            'time_blocks': corridor_risks,
            'stats': {
                'total_incidents': stats['total_incidents'],
                'median_duration': stats['median_duration'],
                'breakdown_pct': round(stats['breakdown_pct'] * 100, 1),
            },
        })

    # Sort by risk
    predictions.sort(key=lambda x: x['overall_probability'], reverse=True)
    return predictions


def generate_shift_brief(predictions, target_date=None, model_accuracy=None):
    """Generate the shift brief document from predictions."""
    if target_date is None:
        target_date = datetime.now() + timedelta(days=1)

    brief = {
        'title': 'BENGALURU TRAFFIC INTELLIGENCE',
        'subtitle': f"SHIFT BRIEF — {target_date.strftime('%d %B %Y').upper()} EARLY MORNING",
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'model_accuracy': model_accuracy or 0.843,
        'corridors': [],
        'ekart_alerts': [],
        'summary': {
            'critical_corridors': 0,
            'high_corridors': 0,
            'total_corridors_monitored': len(predictions),
        }
    }

    # Pre-position recommendations based on risk patterns
    deployment_templates = {
        'vehicle_breakdown': 'Crane pre-position: {junction} by {time}',
        'construction': 'Diversion ready: {alt_corridor} from {time}',
        'general': '{units} patrol unit(s): {junction}',
    }

    # Known junctions per corridor (from ASTrAM data)
    corridor_junctions = {
        'Mysore Road': ['Peenya Junction', 'Kengeri Flyover', 'Nayandahalli'],
        'Bellary Road 1': ['Mekhri Circle', 'Sadashivanagar', 'Hebbal Flyover'],
        'Tumkur Road': ['Jalahalli Cross', 'Goraguntepalya', 'Yeshwanthpur'],
        'Bellary Road 2': ['Hebbal', 'Esteem Mall Junction', 'Yelahanka'],
        'Hosur Road': ['Silk Board', 'Madiwala', 'Electronic City'],
        'ORR North 1': ['Hebbal ORR', 'Nagawara', 'KR Puram'],
        'Old Madras Road': ['Indiranagar', 'Baiyyappanahalli', 'KR Puram'],
        'Magadi Road': ['Chord Road Junction', 'Kamakshipalya', 'Vijayanagar'],
        'ORR East 1': ['Marathahalli', 'Bellandur', 'Sarjapur Junction'],
    }

    for pred in predictions:
        corridor = pred['corridor']
        risk = pred['overall_risk_level']

        if risk in ['CRITICAL', 'HIGH']:
            junctions = corridor_junctions.get(corridor, ['Unknown Junction'])
            peak = pred['peak_risk_window']
            peak_start = int(peak.split(':')[0])

            # Generate deployment recommendations
            recommendations = []
            if pred['stats']['breakdown_pct'] > 50:
                pre_pos_time = (peak_start - 1) % 24
                recommendations.append(
                    f"Crane pre-position: {junctions[0]} by {pre_pos_time:02d}:45"
                )
            recommendations.append(
                f"{'2' if risk == 'CRITICAL' else '1'} patrol unit(s): {junctions[-1]}"
            )
            if pred.get('time_blocks'):
                cascade_block = next(
                    (b for b in pred['time_blocks'] if b.get('cascade_active')), None
                )
                if cascade_block:
                    alt = 'Magadi Road' if corridor != 'Magadi Road' else 'Kanakapura Road'
                    recommendations.append(f"Diversion ready: {alt} from {peak_start:02d}:00")

            # Get drivers from peak block
            peak_block = max(pred['time_blocks'], key=lambda x: x['probability'])
            drivers = peak_block.get('drivers', [])

            entry = {
                'risk_level': risk,
                'corridor': corridor,
                'time_window': peak,
                'probability': pred['overall_probability'],
                'confidence': 'HIGH' if pred['overall_probability'] >= 0.75 else 'MODERATE',
                'drivers': drivers,
                'recommendations': recommendations,
                'historical_context': f"{pred['stats']['total_incidents']} incidents on record",
            }
            brief['corridors'].append(entry)

            if risk == 'CRITICAL':
                brief['summary']['critical_corridors'] += 1
            else:
                brief['summary']['high_corridors'] += 1

            # Ekart alert for critical/high corridors
            avoid_start = (peak_start - 1) % 24
            avoid_end = min(peak_start + 3, 23)
            brief['ekart_alerts'].append({
                'corridor': corridor,
                'avoid_window': f"{avoid_start:02d}:30-{avoid_end:02d}:00",
                'estimated_delay_min': int(pred['stats']['median_duration'] or 60),
                'risk_level': risk,
                'alternate': 'Kanakapura Road' if corridor == 'Mysore Road' else 'NICE Road',
                'alternate_risk': 'LOW',
            })

    return brief


if __name__ == "__main__":
    artifacts = load_model()
    print("Model loaded successfully")

    predictions = generate_risk_scores(artifacts)
    print(f"\nGenerated predictions for {len(predictions)} corridors")

    for pred in predictions[:5]:
        print(f"\n{pred['overall_risk_level']:10s} {pred['corridor']:20s} "
              f"P={pred['overall_probability']:.2f} Window={pred['peak_risk_window']}")

    brief = generate_shift_brief(predictions, model_accuracy=artifacts['metrics']['precision'])
    print(f"\nShift Brief generated: {brief['summary']['critical_corridors']} critical, "
          f"{brief['summary']['high_corridors']} high risk corridors")
