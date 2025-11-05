#!/usr/bin/env python3
"""
AWS Spot Optimizer v7.3 + Holiday Validator - COMPLETE
=======================================================

MODIFICATION: Added Holiday Validator as pre-processing step
- Step 1: Holiday Validator checks calendar dates against historical data
- Step 2: Prophet uses validated scores for training
- Step 3-5: Same as original v7.3 (Risk, Decision, Backtest)

Pipeline:
┌─────────────────────────────────────────┐
│ STEP 1: Holiday Validator (NEW!)       │
│ ├─ Takes: Holiday calendar dates       │
│ ├─ Checks: Historical price data       │
│ ├─ Calculates: Real impact scores      │
│ └─ Outputs: Validated scores            │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ STEP 2: Prophet Training (v7.3)        │
│ ├─ Uses: Validated holiday scores      │
│ └─ Outputs: Trained models              │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ STEP 3: Risk Scoring (v7.3)            │
│ ├─ Uses: Validated scores in risk calc │
│ └─ Outputs: Risk profiles               │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ STEP 4: Decision Engine (v7.3)         │
│ └─ Outputs: Optimal decisions           │
└─────────────────────────────────────────┘
           ↓
┌─────────────────────────────────────────┐
│ STEP 5: Backtest (v7.3)                │
│ └─ Outputs: Costs, savings, metrics     │
└─────────────────────────────────────────┘

Version: 7.3 + Holiday Validator
Date: 2025-11-05
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
import multiprocessing as mp
from prophet import Prophet
from datetime import datetime, timedelta
from tqdm.auto import tqdm
import json
import os
import pickle
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Production configuration"""
    
    # Hardware
    USE_GPU = torch.backends.mps.is_available()
    N_CORES = mp.cpu_count()
    DEVICE = torch.device("mps" if USE_GPU else "cpu")
    
    # Paths
    TRAINING_DATA = '/Users/atharvapudale/Downloads/aws_2023_2024_complete_24months.csv'
    TEST_DATA = '/Users/atharvapudale/Downloads/mumbai_spot_data_sorted_asc(1-2-3-25).csv'
    EVENT_DATA = '/Users/atharvapudale/Downloads/aws_stress_events_2023_2025.csv'
    OUTPUT_DIR = './outputs/hybrid_v7_3_holiday_validated'
    MODEL_DIR = './models/v7_3_validated'
    
    INSTANCE_TYPES = ['t3.medium', 't4g.small', 't4g.medium', 'c5.large']
    
    # Prophet Settings
    PROPHET_CHANGEPOINT_PRIOR = 0.001
    PROPHET_SEASONALITY_PRIOR = 0.01
    PROPHET_INTERVAL_WIDTH = 0.95
    PROPHET_DAILY_SEASONALITY = True
    PROPHET_WEEKLY_SEASONALITY = True
    PROPHET_YEARLY_SEASONALITY = False
    
    # Risk Weights
    RISK_WEIGHT_CAPACITY = 0.30
    RISK_WEIGHT_VOLATILITY = 0.15
    RISK_WEIGHT_TREND = 0.20
    RISK_WEIGHT_EVENT = 0.30  # Events properly weighted
    RISK_WEIGHT_PREDICTION = 0.05
    
    # Capacity Thresholds
    CAPACITY_EXCELLENT = 0.30
    CAPACITY_GOOD = 0.35
    CAPACITY_MODERATE = 0.42
    CAPACITY_WARNING = 0.48
    CAPACITY_CRITICAL = 0.52
    
    # Risk Categories
    RISK_SAFE_MAX = 30
    RISK_LOW_MAX = 45
    RISK_MODERATE_MAX = 60
    
    # Decision Rules
    SWITCH_MIN_SAVINGS = 0.03
    SWITCH_RISK_DIFF_MAX = 15
    SWITCH_RISK_ABSOLUTE_MAX = 65
    ONDEMAND_RISK_THRESHOLD = 45
    
    # Event Parameters
    EVENT_CRITICAL_IMPACT = 70
    EVENT_PREEMPTIVE_DAYS = 3
    EVENT_POSTMORTEM_DAYS = 2
    
    # Holiday Validator Parameters (NEW!)
    HOLIDAY_VALIDATION_WINDOW = 5  # ±5 days around holiday to check
    
    # Other
    DECISION_INTERVAL_HOURS = 12
    FORECAST_HORIZON_HOURS = 168
    MAX_POOL_AGE_DAYS = 21
    DIVERSITY_MIN_SAVINGS = 0.02
    
    # Concentration Risk Penalties
    SINGLE_POOL_CONCENTRATION_PENALTY = 20
    SINGLE_POOL_NO_ESCAPE_PENALTY = 10

config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)

print("="*80)
print("AWS SPOT OPTIMIZER V7.3 + HOLIDAY VALIDATOR")
print("="*80)
print(f"\n⚙️  Hardware: {config.DEVICE}, {config.N_CORES} cores")
print(f"🎯 Pipeline:")
print(f"   Step 1: Holiday Validator → Validates calendar against historical data")
print(f"   Step 2: Prophet Training → Uses validated scores")
print(f"   Step 3: Risk Scoring → Event-aware with validated scores")
print(f"   Step 4: Decision Engine → Optimal pool selection")
print(f"   Step 5: Backtest → Performance metrics")
print("\n" + "="*80 + "\n")

# ============================================================================
# STEP 1: HOLIDAY VALIDATOR (NEW!)
# ============================================================================

class HolidayValidator:
    """Validates holiday calendar against historical price data"""
    
    def __init__(self):
        self.validated_holidays = {}
        self.baseline_stats = {}
    
    def validate_holidays(self, df_train, holiday_csv_path):
        """
        Takes holiday dates from calendar, validates against historical data,
        generates data-driven severity scores
        """
        print("\n" + "="*80)
        print("🔍 STEP 1: HOLIDAY VALIDATION")
        print("="*80)
        
        # Calculate baseline statistics from historical data
        print("\n📊 Calculating baseline from historical data...")
        df_train['date'] = df_train['timestamp'].dt.date
        daily_stats = df_train.groupby('date').agg({
            'SpotPrice': ['mean', 'std', 'max'],
            'ratio': ['mean', 'std', 'max']
        }).reset_index()
        daily_stats.columns = ['date', 'price_mean', 'price_std', 'price_max',
                               'ratio_mean', 'ratio_std', 'ratio_max']
        
        self.baseline_stats = {
            'price_median': daily_stats['price_mean'].median(),
            'price_std': daily_stats['price_std'].median(),
            'ratio_median': daily_stats['ratio_mean'].median(),
            'ratio_std': daily_stats['ratio_std'].median()
        }
        
        print(f"   Baseline price: ${self.baseline_stats['price_median']:.4f}")
        print(f"   Baseline ratio: {self.baseline_stats['ratio_median']:.3f}")
        print(f"   Baseline volatility: ${self.baseline_stats['price_std']:.4f}")
        
        # Load holiday calendar
        print(f"\n📅 Loading holiday calendar from: {holiday_csv_path}")
        df_holidays = pd.read_csv(holiday_csv_path)
        df_holidays.columns = df_holidays.columns.str.lower().str.strip()
        
        date_col = next((c for c in df_holidays.columns if 'date' in c), None)
        name_col = next((c for c in df_holidays.columns if 'event' in c or 'name' in c), None)
        impact_col = next((c for c in df_holidays.columns if 'impact' in c), None)
        region_col = next((c for c in df_holidays.columns if 'region' in c), None)
        
        if not date_col or not name_col:
            print("   ⚠️  Could not find required columns (date, name)")
            return
        
        df_holidays[date_col] = pd.to_datetime(df_holidays[date_col])
        
        print(f"   Loaded {len(df_holidays)} holidays from calendar")
        
        # Validate each holiday against historical data
        print(f"\n🧪 Validating holidays against historical data...")
        print(f"   Checking ±{config.HOLIDAY_VALIDATION_WINDOW} days around each holiday")
        
        train_start = df_train['timestamp'].min()
        train_end = df_train['timestamp'].max()
        
        validated_count = 0
        not_in_training = 0
        insufficient_data = 0
        
        claimed_scores_map = {'Critical': 100, 'High': 75, 'Medium': 50, 'Low': 25}
        
        for _, holiday in tqdm(df_holidays.iterrows(), total=len(df_holidays), desc="Validating"):
            holiday_date = pd.Timestamp(holiday[date_col])
            holiday_name = str(holiday[name_col]) if name_col else 'Holiday'
            claimed_impact = str(holiday[impact_col]).strip() if impact_col else 'Medium'
            claimed_score = claimed_scores_map.get(claimed_impact, 50)
            holiday_region = str(holiday[region_col]) if region_col else 'all'
            
            # Check if holiday is in training period
            if not (train_start <= holiday_date <= train_end):
                # Future holiday - use claimed score
                self.validated_holidays[f"{holiday_name}_{holiday_date.date()}"] = {
                    'date': holiday_date,
                    'name': holiday_name,
                    'claimed_impact': claimed_impact,
                    'claimed_score': claimed_score,
                    'validated_score': claimed_score,
                    'region': holiday_region,
                    'status': 'NOT_IN_TRAINING_PERIOD'
                }
                not_in_training += 1
                continue
            
            # Extract data around holiday (±5 days)
            window_start = holiday_date - timedelta(days=config.HOLIDAY_VALIDATION_WINDOW)
            window_end = holiday_date + timedelta(days=config.HOLIDAY_VALIDATION_WINDOW)
            
            holiday_data = df_train[
                (df_train['timestamp'] >= window_start) &
                (df_train['timestamp'] <= window_end)
            ]
            
            if len(holiday_data) < 100:
                # Insufficient data - use claimed score
                self.validated_holidays[f"{holiday_name}_{holiday_date.date()}"] = {
                    'date': holiday_date,
                    'name': holiday_name,
                    'claimed_impact': claimed_impact,
                    'claimed_score': claimed_score,
                    'validated_score': claimed_score,
                    'region': holiday_region,
                    'status': 'INSUFFICIENT_DATA'
                }
                insufficient_data += 1
                continue
            
            # Calculate impact metrics from actual data
            price_spike_pct = ((holiday_data['SpotPrice'].max() - self.baseline_stats['price_median']) /
                              self.baseline_stats['price_median'] * 100) if self.baseline_stats['price_median'] > 0 else 0
            
            ratio_increase_pct = ((holiday_data['ratio'].mean() - self.baseline_stats['ratio_median']) /
                                 self.baseline_stats['ratio_median'] * 100) if self.baseline_stats['ratio_median'] > 0 else 0
            
            volatility_increase_pct = ((holiday_data['SpotPrice'].std() - self.baseline_stats['price_std']) /
                                      self.baseline_stats['price_std'] * 100) if self.baseline_stats['price_std'] > 0 else 0
            
            # Calculate data-driven validated score (0-100)
            validated_score = 0
            
            # Price spike component (40% weight)
            if price_spike_pct > 100: validated_score += 40
            elif price_spike_pct > 50: validated_score += 30
            elif price_spike_pct > 25: validated_score += 20
            elif price_spike_pct > 10: validated_score += 10
            
            # Capacity ratio component (40% weight)
            if ratio_increase_pct > 30: validated_score += 40
            elif ratio_increase_pct > 20: validated_score += 30
            elif ratio_increase_pct > 10: validated_score += 20
            elif ratio_increase_pct > 5: validated_score += 10
            
            # Volatility component (20% weight)
            if volatility_increase_pct > 50: validated_score += 20
            elif volatility_increase_pct > 30: validated_score += 15
            elif volatility_increase_pct > 15: validated_score += 10
            elif volatility_increase_pct > 5: validated_score += 5
            
            # Store validated holiday
            self.validated_holidays[f"{holiday_name}_{holiday_date.date()}"] = {
                'date': holiday_date,
                'name': holiday_name,
                'claimed_impact': claimed_impact,
                'claimed_score': claimed_score,
                'validated_score': validated_score,
                'price_spike_pct': price_spike_pct,
                'ratio_increase_pct': ratio_increase_pct,
                'volatility_increase_pct': volatility_increase_pct,
                'region': holiday_region,
                'status': 'VALIDATED',
                'data_points': len(holiday_data)
            }
            
            validated_count += 1
        
        print(f"\n✅ Validation Complete:")
        print(f"   Validated from data: {validated_count}")
        print(f"   Not in training period: {not_in_training}")
        print(f"   Insufficient data: {insufficient_data}")
        print(f"   Total: {len(self.validated_holidays)}")
        
        # Print validation summary
        validated_list = [h for h in self.validated_holidays.values() if h['status'] == 'VALIDATED']
        
        if validated_list:
            avg_diff = np.mean([h['validated_score'] - h['claimed_score'] for h in validated_list])
            print(f"\n📊 Validation Summary (for {len(validated_list)} validated holidays):")
            print(f"   Average score difference: {avg_diff:+.1f} points")
            print(f"   (Negative = calendar overestimated, Positive = calendar underestimated)")
            
            overestimated = len([h for h in validated_list if h['validated_score'] < h['claimed_score'] - 20])
            underestimated = len([h for h in validated_list if h['validated_score'] > h['claimed_score'] + 20])
            accurate = len([h for h in validated_list if abs(h['validated_score'] - h['claimed_score']) <= 20])
            
            print(f"   Significantly overestimated: {overestimated}")
            print(f"   Reasonably accurate: {accurate}")
            print(f"   Significantly underestimated: {underestimated}")
            
            # Show top 5 validated holidays by score
            print(f"\n   🔝 Top 5 Validated Holidays (by impact):")
            sorted_holidays = sorted(validated_list, key=lambda x: x['validated_score'], reverse=True)
            for i, h in enumerate(sorted_holidays[:5], 1):
                diff = h['validated_score'] - h['claimed_score']
                status = "✅" if abs(diff) < 20 else ("⚠️ OVER" if diff < 0 else "⚠️ UNDER")
                print(f"   {i}. {h['date'].date()} - {h['name']}")
                print(f"      Claimed: {h['claimed_score']} → Validated: {h['validated_score']} {status} ({diff:+.0f})")
    
    def get_holiday_score(self, holiday_name, holiday_date, default=50):
        """Get validated score for a specific holiday"""
        key = f"{holiday_name}_{holiday_date.date()}"
        if key in self.validated_holidays:
            return self.validated_holidays[key]['validated_score']
        
        # Fallback: search by name only
        for h_key, h_info in self.validated_holidays.items():
            if h_info['name'] == holiday_name:
                return h_info['validated_score']
        
        return default
    
    def save(self):
        """Save validated holidays to disk"""
        save_data = {
            'baseline_stats': self.baseline_stats,
            'validated_holidays': {
                key: {**info, 'date': info['date'].isoformat()}
                for key, info in self.validated_holidays.items()
            }
        }
        
        path = os.path.join(config.MODEL_DIR, 'validated_holidays.json')
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n💾 Saved validated holidays: {path}")
    
    def load(self):
        """Load previously validated holidays"""
        path = os.path.join(config.MODEL_DIR, 'validated_holidays.json')
        
        if os.path.exists(path):
            print("\n📂 Loading validated holidays...")
            with open(path) as f:
                save_data = json.load(f)
            
            self.baseline_stats = save_data['baseline_stats']
            self.validated_holidays = {
                key: {**info, 'date': pd.Timestamp(info['date'])}
                for key, info in save_data['validated_holidays'].items()
            }
            
            validated_count = len([h for h in self.validated_holidays.values() if h['status'] == 'VALIDATED'])
            print(f"   ✅ Loaded {len(self.validated_holidays)} holidays ({validated_count} validated from data)")
            return True
        
        return False

# ============================================================================
# EVENT CALENDAR (Uses validated scores - MODIFIED)
# ============================================================================

class EventCalendar:
    def __init__(self, event_csv_path, holiday_validator):
        self.events = []
        self.holiday_validator = holiday_validator
        self.load_events(event_csv_path)
    
    def load_events(self, path):
        print("\n" + "="*80)
        print("📅 STEP 2: Loading Event Calendar with Validated Scores")
        print("="*80)
        
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.strip()
        
        date_col = next((c for c in df.columns if 'date' in c), None)
        name_col = next((c for c in df.columns if 'event' in c or 'name' in c), None)
        impact_col = next((c for c in df.columns if 'impact' in c), None)
        region_col = next((c for c in df.columns if 'region' in c), None)
        
        if not date_col:
            print("   ⚠️  No date column found")
            return
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        for _, row in df.iterrows():
            event_name = str(row[name_col]) if name_col else 'Event'
            event_date = pd.Timestamp(row[date_col])
            
            # Get VALIDATED score (this is the key change!)
            validated_score = self.holiday_validator.get_holiday_score(event_name, event_date, default=50)
            
            self.events.append({
                'date': event_date,
                'name': event_name,
                'impact_score': validated_score,  # USING VALIDATED SCORE!
                'region': str(row[region_col]) if region_col else 'all'
            })
        
        print(f"   ✅ Loaded {len(self.events)} events with validated scores")
        
        # Count how many use validated scores
        validated_count = 0
        for event in self.events:
            key = f"{event['name']}_{event['date'].date()}"
            if key in self.holiday_validator.validated_holidays:
                if self.holiday_validator.validated_holidays[key]['status'] == 'VALIDATED':
                    validated_count += 1
        
        print(f"   🧠 {validated_count} events using data-validated scores")
        print(f"   📝 {len(self.events) - validated_count} events using default/claimed scores")
    
    def check_event_window(self, current_date, region='all'):
        if not isinstance(current_date, pd.Timestamp):
            current_date = pd.Timestamp(current_date)
        
        max_impact = 0
        relevant_event = None
        
        for event in self.events:
            if event['region'] not in ['all', region]:
                continue
            
            event_start = event['date'] - timedelta(days=config.EVENT_PREEMPTIVE_DAYS)
            event_end = event['date'] + timedelta(days=config.EVENT_POSTMORTEM_DAYS)
            
            if event_start <= current_date <= event_end:
                if event['impact_score'] > max_impact:
                    max_impact = event['impact_score']
                    relevant_event = event
        
        return max_impact, relevant_event

# ============================================================================
# DATA LOADER (Same as v7.3)
# ============================================================================

def load_and_clean(csv_path, name):
    print(f"\n{'='*80}")
    print(f"📥 Loading {name}")
    print(f"{'='*80}")
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df):,} records")
    
    df.columns = df.columns.str.lower().str.strip()
    col_map = {}
    for col in df.columns:
        if 'time' in col or 'date' in col:
            col_map[col] = 'timestamp'
        elif 'spot' in col and 'price' in col:
            col_map[col] = 'SpotPrice'
        elif 'ondemand' in col or 'on_demand' in col:
            col_map[col] = 'OnDemandPrice'
        elif 'instance' in col and 'type' in col:
            col_map[col] = 'InstanceType'
        elif col in ['az', 'availability_zone']:
            col_map[col] = 'AZ'
        elif 'region' in col:
            col_map[col] = 'Region'
    
    df = df.rename(columns=col_map)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print("🧹 Cleaning...")
    before = len(df)
    df = df[(df['SpotPrice'] >= 0) & (df['SpotPrice'] < 10) & (df['OnDemandPrice'] > 0)].copy()
    removed = before - len(df)
    if removed > 0:
        print(f"   Removed {removed:,} corrupt records ({removed/before*100:.2f}%)")
    
    df = df[df['InstanceType'].isin(config.INSTANCE_TYPES)].copy()
    
    if 'Region' not in df.columns:
        df['Region'] = df['AZ'].str.extract(r'^([a-z]+-[a-z]+-\d+)')[0].fillna('ap-south-1')
    
    df['Pool_ID'] = df['InstanceType'] + '_' + df['AZ']
    df['ratio'] = (df['SpotPrice'] / df['OnDemandPrice']).clip(0, 1)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ Clean: {len(df):,} records, {df['Pool_ID'].nunique()} pools")
    print(f"   Dates: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"   Avg Ratio: {df['ratio'].mean():.3f} ({(1-df['ratio'].mean())*100:.1f}% savings)")
    
    return df

# ============================================================================
# RISK SCORER (Same as v7.3)
# ============================================================================

class RiskScorer:
    def __init__(self, event_calendar):
        self.event_calendar = event_calendar
        self.risk_history = defaultdict(list)
    
    def compute_capacity_risk(self, pool_data):
        current_ratio = pool_data['ratio'].iloc[-1] if len(pool_data) > 0 else 0.5
        
        if current_ratio < config.CAPACITY_EXCELLENT:
            return 0
        elif current_ratio < config.CAPACITY_GOOD:
            return 20
        elif current_ratio < config.CAPACITY_MODERATE:
            return 40
        elif current_ratio < config.CAPACITY_WARNING:
            return 65
        else:
            return 90
    
    def compute_volatility_risk(self, pool_data):
        if len(pool_data) < 24:
            return 30
        recent = pool_data.tail(168)
        mean_price = recent['SpotPrice'].mean()
        std_price = recent['SpotPrice'].std()
        cv = (std_price / mean_price) if mean_price > 0 else 0
        risk = (cv / 0.15) * 100
        return np.clip(risk, 0, 100)
    
    def compute_trend_risk(self, pool_data):
        if len(pool_data) < 48:
            return 0
        recent_24h = pool_data.tail(24)['ratio'].mean()
        previous_24h = pool_data.tail(48).head(24)['ratio'].mean()
        ratio_change = (recent_24h - previous_24h) / (previous_24h + 1e-6)
        risk = (ratio_change / 0.10) * 100
        return np.clip(risk, 0, 100)
    
    def compute_event_risk(self, current_date, region):
        # Uses VALIDATED scores from event calendar
        impact_score, event = self.event_calendar.check_event_window(current_date, region)
        if impact_score >= config.EVENT_CRITICAL_IMPACT:
            return impact_score
        return impact_score * 0.5
    
    def compute_prediction_risk(self, pool_data, has_model):
        if not has_model:
            return 50
        if len(pool_data) < 168:
            return 30
        last_timestamp = pool_data['timestamp'].iloc[-1]
        hours_old = (datetime.now() - last_timestamp).total_seconds() / 3600
        risk = (hours_old / 24) * 100
        return np.clip(risk, 0, 100)
    
    def compute_overall_risk(self, pool_id, pool_data, current_date, has_model=True, is_single_pool_strategy=False):
        region = pool_data['Region'].iloc[0] if len(pool_data) > 0 else 'ap-south-1'
        
        capacity_risk = self.compute_capacity_risk(pool_data)
        volatility_risk = self.compute_volatility_risk(pool_data)
        trend_risk = self.compute_trend_risk(pool_data)
        event_risk = self.compute_event_risk(current_date, region)
        prediction_risk = self.compute_prediction_risk(pool_data, has_model)
        
        base_risk = (
            config.RISK_WEIGHT_CAPACITY * capacity_risk +
            config.RISK_WEIGHT_VOLATILITY * volatility_risk +
            config.RISK_WEIGHT_TREND * trend_risk +
            config.RISK_WEIGHT_EVENT * event_risk +
            config.RISK_WEIGHT_PREDICTION * prediction_risk
        )
        
        overall_risk = base_risk
        concentration_penalty = 0
        
        if is_single_pool_strategy:
            concentration_penalty += config.SINGLE_POOL_CONCENTRATION_PENALTY
            concentration_penalty += config.SINGLE_POOL_NO_ESCAPE_PENALTY
            
            current_ratio = pool_data['ratio'].iloc[-1] if len(pool_data) > 0 else 0.5
            if current_ratio > 0.45:
                concentration_penalty += 5
            
            overall_risk = min(base_risk + concentration_penalty, 100)
        
        if overall_risk < config.RISK_SAFE_MAX:
            category = 'SAFE'
        elif overall_risk < config.RISK_LOW_MAX:
            category = 'LOW'
        elif overall_risk < config.RISK_MODERATE_MAX:
            category = 'MODERATE'
        else:
            category = 'HIGH'
        
        risk_profile = {
            'pool_id': pool_id,
            'overall_risk': overall_risk,
            'base_risk': base_risk,
            'concentration_penalty': concentration_penalty,
            'capacity_risk': capacity_risk,
            'volatility_risk': volatility_risk,
            'trend_risk': trend_risk,
            'event_risk': event_risk,
            'prediction_risk': prediction_risk,
            'risk_category': category,
            'current_ratio': pool_data['ratio'].iloc[-1] if len(pool_data) > 0 else 0.5,
            'timestamp': current_date
        }
        
        self.risk_history[pool_id].append(risk_profile)
        return risk_profile

# ============================================================================
# PROPHET FORECASTER (Same as v7.3)
# ============================================================================

class ProphetForecaster:
    def __init__(self, event_calendar):
        self.event_calendar = event_calendar
        self.models = {}
        self.metrics = {}
    
    def prepare_holidays(self, pool_data, region):
        holidays_list = []
        date_range = pd.date_range(
            start=pool_data['timestamp'].min(),
            end=pool_data['timestamp'].max(),
            freq='D'
        )
        
        for date in date_range:
            # Uses VALIDATED scores
            impact_score, event = self.event_calendar.check_event_window(date, region)
            if impact_score > 50:
                holidays_list.append({
                    'ds': date,
                    'holiday': event['name'] if event else 'Event',
                    'lower_window': -config.EVENT_PREEMPTIVE_DAYS,
                    'upper_window': config.EVENT_POSTMORTEM_DAYS
                })
        
        if holidays_list:
            return pd.DataFrame(holidays_list).drop_duplicates(subset=['ds'])
        return None
    
    def train_pool(self, pool_id, pool_data):
        try:
            prophet_df = pool_data.groupby('timestamp').agg({'SpotPrice': 'mean'}).reset_index()
            prophet_df.columns = ['ds', 'y']
            
            if len(prophet_df) < 200:
                return None, {'error': 'Insufficient data'}
            
            region = pool_data['Region'].iloc[0]
            holidays_df = self.prepare_holidays(pool_data, region)
            
            model = Prophet(
                changepoint_prior_scale=config.PROPHET_CHANGEPOINT_PRIOR,
                seasonality_prior_scale=config.PROPHET_SEASONALITY_PRIOR,
                interval_width=config.PROPHET_INTERVAL_WIDTH,
                daily_seasonality=config.PROPHET_DAILY_SEASONALITY,
                weekly_seasonality=config.PROPHET_WEEKLY_SEASONALITY,
                yearly_seasonality=config.PROPHET_YEARLY_SEASONALITY,
                holidays=holidays_df
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(prophet_df)
            
            train_df = prophet_df[:-168]
            val_df = prophet_df[-168:]
            
            if len(val_df) > 0:
                model_val = Prophet(
                    changepoint_prior_scale=config.PROPHET_CHANGEPOINT_PRIOR,
                    seasonality_prior_scale=config.PROPHET_SEASONALITY_PRIOR,
                    interval_width=config.PROPHET_INTERVAL_WIDTH,
                    daily_seasonality=config.PROPHET_DAILY_SEASONALITY,
                    weekly_seasonality=config.PROPHET_WEEKLY_SEASONALITY,
                    yearly_seasonality=config.PROPHET_YEARLY_SEASONALITY,
                    holidays=holidays_df
                )
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model_val.fit(train_df)
                
                val_future = pd.DataFrame({'ds': val_df['ds']})
                val_forecast = model_val.predict(val_future)
                val_actual = val_df['y'].values
                val_pred = val_forecast['yhat'].values[:len(val_actual)]
                mape = np.mean(np.abs((val_actual - val_pred) / (val_actual + 1e-6))) * 100
                metrics = {'mape': mape, 'records': len(prophet_df)}
            else:
                metrics = {'mape': 0, 'records': len(prophet_df)}
            
            return model, metrics
        except Exception as e:
            return None, {'error': str(e)}
    
    def train_all(self, df_train):
        print("\n" + "="*80)
        print("🤖 STEP 3: Training Prophet Models with Validated Holiday Scores")
        print("="*80)
        
        pools = df_train['Pool_ID'].unique()
        print(f"Training {len(pools)} pool models...")
        
        successful = 0
        pbar = tqdm(pools, desc="Training")
        
        for pool_id in pbar:
            pool_data = df_train[df_train['Pool_ID'] == pool_id].copy()
            model, metrics = self.train_pool(pool_id, pool_data)
            
            if model is not None:
                self.models[pool_id] = model
                self.metrics[pool_id] = metrics
                successful += 1
                pbar.set_postfix({'success': successful})
        
        print(f"\n✅ Trained {successful}/{len(pools)} models with validated holiday effects")
        return successful
    
    def save_models(self):
        print("\n💾 Saving Prophet models...")
        model_path = os.path.join(config.MODEL_DIR, 'prophet_models.pkl')
        metrics_path = os.path.join(config.MODEL_DIR, 'prophet_metrics.json')
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.models, f)
        
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"   ✅ Saved {len(self.models)} models")
    
    def load_models(self):
        model_path = os.path.join(config.MODEL_DIR, 'prophet_models.pkl')
        metrics_path = os.path.join(config.MODEL_DIR, 'prophet_metrics.json')
        
        if os.path.exists(model_path):
            print("\n📂 Loading saved Prophet models...")
            with open(model_path, 'rb') as f:
                self.models = pickle.load(f)
            
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    self.metrics = json.load(f)
            
            print(f"   ✅ Loaded {len(self.models)} models")
            return True
        return False
    
    def predict(self, pool_id, last_timestamp, periods=168):
        if pool_id not in self.models:
            return None
        
        model = self.models[pool_id]
        future_dates = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=periods,
            freq='H'
        )
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = model.predict(future_df)
        
        return pd.DataFrame({
            'timestamp': future_dates,
            'predicted_price': forecast['yhat'].values,
            'lower_bound': forecast['yhat_lower'].values,
            'upper_bound': forecast['yhat_upper'].values
        })

# [CONTINUING IN NEXT MESSAGE - Decision Engine, Backtester, Main - same as v7.3]


# ============================================================================
# DECISION ENGINE (Same as v7.3)
# ============================================================================

class DecisionEngine:
    def __init__(self, forecaster, risk_scorer):
        self.forecaster = forecaster
        self.risk_scorer = risk_scorer
        self.current_pool = {}
        self.current_mode = {}
        self.pool_entry_time = {}
        self.decisions = []
    
    def evaluate_pool(self, pool_id, pool_data, current_time):
        current_price = pool_data['SpotPrice'].iloc[-1] if len(pool_data) > 0 else 999
        has_model = pool_id in self.forecaster.models
        risk_profile = self.risk_scorer.compute_overall_risk(
            pool_id, pool_data, current_time, has_model, is_single_pool_strategy=False
        )
        
        forecast_price = current_price
        if has_model:
            forecast = self.forecaster.predict(pool_id, pool_data['timestamp'].iloc[-1], periods=24)
            if forecast is not None:
                forecast_price = forecast['predicted_price'].mean()
        
        return {
            'pool_id': pool_id,
            'current_price': current_price,
            'forecast_price': forecast_price,
            'risk_profile': risk_profile,
            'ondemand_price': pool_data['OnDemandPrice'].iloc[-1] if len(pool_data) > 0 else 999
        }
    
    def make_decision(self, instance_type, all_pools_data, current_time):
        # Initialize
        if instance_type not in self.current_pool:
            evaluations = []
            for pool_id in all_pools_data['Pool_ID'].unique():
                pool_data = all_pools_data[all_pools_data['Pool_ID'] == pool_id]
                eval_result = self.evaluate_pool(pool_id, pool_data, current_time)
                evaluations.append(eval_result)
            
            if not evaluations:
                return {'action': 'ERROR', 'reason': 'No pools available'}
            
            evaluations.sort(key=lambda x: (x['risk_profile']['overall_risk'], x['current_price']))
            best = evaluations[0]
            
            self.current_pool[instance_type] = best['pool_id']
            self.current_mode[instance_type] = 'SPOT'
            self.pool_entry_time[instance_type] = current_time
            
            return {
                'action': 'INIT',
                'mode': 'SPOT',
                'target': best['pool_id'],
                'risk': best['risk_profile']['overall_risk'],
                'price': best['current_price']
            }
        
        # Current pool
        current_pool_id = self.current_pool[instance_type]
        current_mode = self.current_mode[instance_type]
        
        current_pool_data = all_pools_data[all_pools_data['Pool_ID'] == current_pool_id]
        if len(current_pool_data) == 0:
            return {'action': 'HOLD', 'mode': current_mode, 'target': current_pool_id}
        
        current_eval = self.evaluate_pool(current_pool_id, current_pool_data, current_time)
        current_risk = current_eval['risk_profile']['overall_risk']
        current_price = current_eval['current_price']
        ondemand_price = current_eval['ondemand_price']
        
        # High Risk → On-Demand
        if current_risk > config.ONDEMAND_RISK_THRESHOLD:
            if current_mode != 'ONDEMAND':
                self.current_mode[instance_type] = 'ONDEMAND'
                return {
                    'action': 'SWITCH_TO_ONDEMAND',
                    'mode': 'ONDEMAND',
                    'target': 'ON_DEMAND',
                    'risk': 0,
                    'price': ondemand_price
                }
            else:
                if current_risk < config.RISK_MODERATE_MAX:
                    evaluations = []
                    for pool_id in all_pools_data['Pool_ID'].unique():
                        pool_data = all_pools_data[all_pools_data['Pool_ID'] == pool_id]
                        eval_result = self.evaluate_pool(pool_id, pool_data, current_time)
                        if eval_result['risk_profile']['overall_risk'] < config.RISK_MODERATE_MAX:
                            evaluations.append(eval_result)
                    
                    if evaluations:
                        evaluations.sort(key=lambda x: (x['risk_profile']['overall_risk'], x['current_price']))
                        best = evaluations[0]
                        
                        self.current_pool[instance_type] = best['pool_id']
                        self.current_mode[instance_type] = 'SPOT'
                        self.pool_entry_time[instance_type] = current_time
                        
                        return {
                            'action': 'RETURN_TO_SPOT',
                            'mode': 'SPOT',
                            'target': best['pool_id'],
                            'risk': best['risk_profile']['overall_risk'],
                            'price': best['current_price']
                        }
                
                return {
                    'action': 'HOLD',
                    'mode': 'ONDEMAND',
                    'target': 'ON_DEMAND',
                    'risk': 0,
                    'price': ondemand_price
                }
        
        # On-Demand but safe now
        if current_mode == 'ONDEMAND' and current_risk < config.RISK_MODERATE_MAX:
            evaluations = []
            for pool_id in all_pools_data['Pool_ID'].unique():
                pool_data = all_pools_data[all_pools_data['Pool_ID'] == pool_id]
                eval_result = self.evaluate_pool(pool_id, pool_data, current_time)
                if eval_result['risk_profile']['overall_risk'] < config.RISK_MODERATE_MAX:
                    evaluations.append(eval_result)
            
            if evaluations:
                evaluations.sort(key=lambda x: (x['risk_profile']['overall_risk'], x['current_price']))
                best = evaluations[0]
                
                self.current_pool[instance_type] = best['pool_id']
                self.current_mode[instance_type] = 'SPOT'
                self.pool_entry_time[instance_type] = current_time
                
                return {
                    'action': 'SWITCH_TO_SPOT',
                    'mode': 'SPOT',
                    'target': best['pool_id'],
                    'risk': best['risk_profile']['overall_risk'],
                    'price': best['current_price']
                }
        
        # Find better Spot pool
        if current_mode == 'SPOT':
            candidates = []
            
            for pool_id in all_pools_data['Pool_ID'].unique():
                if pool_id == current_pool_id:
                    continue
                
                pool_data = all_pools_data[all_pools_data['Pool_ID'] == pool_id]
                eval_result = self.evaluate_pool(pool_id, pool_data, current_time)
                
                candidate_risk = eval_result['risk_profile']['overall_risk']
                candidate_price = eval_result['current_price']
                
                if candidate_risk > config.SWITCH_RISK_ABSOLUTE_MAX:
                    continue
                
                risk_increase = candidate_risk - current_risk
                if risk_increase > config.SWITCH_RISK_DIFF_MAX:
                    continue
                
                savings_pct = (current_price - candidate_price) / current_price
                
                if savings_pct >= config.SWITCH_MIN_SAVINGS:
                    eval_result['savings_pct'] = savings_pct
                    eval_result['switch_score'] = savings_pct * 100 - risk_increase
                    candidates.append(eval_result)
                elif candidate_risk < current_risk - 10:
                    eval_result['savings_pct'] = savings_pct
                    eval_result['switch_score'] = (current_risk - candidate_risk) * 2
                    candidates.append(eval_result)
            
            if candidates:
                candidates.sort(key=lambda x: -x['switch_score'])
                best = candidates[0]
                
                self.current_pool[instance_type] = best['pool_id']
                self.pool_entry_time[instance_type] = current_time
                
                return {
                    'action': 'SWITCH_SPOT_POOL',
                    'mode': 'SPOT',
                    'target': best['pool_id'],
                    'risk': best['risk_profile']['overall_risk'],
                    'price': best['current_price']
                }
        
        # Diversity
        if instance_type in self.pool_entry_time:
            days_in_pool = (current_time - self.pool_entry_time[instance_type]).days
            
            if days_in_pool >= config.MAX_POOL_AGE_DAYS:
                evaluations = []
                for pool_id in all_pools_data['Pool_ID'].unique():
                    if pool_id == current_pool_id:
                        continue
                    
                    pool_data = all_pools_data[all_pools_data['Pool_ID'] == pool_id]
                    eval_result = self.evaluate_pool(pool_id, pool_data, current_time)
                    
                    if eval_result['risk_profile']['overall_risk'] < config.SWITCH_RISK_ABSOLUTE_MAX:
                        savings_pct = (current_price - eval_result['current_price']) / current_price
                        if savings_pct >= config.DIVERSITY_MIN_SAVINGS:
                            evaluations.append(eval_result)
                
                if evaluations:
                    evaluations.sort(key=lambda x: (x['risk_profile']['overall_risk'], x['current_price']))
                    best = evaluations[0]
                    
                    self.current_pool[instance_type] = best['pool_id']
                    self.pool_entry_time[instance_type] = current_time
                    
                    return {
                        'action': 'DIVERSITY_SWITCH',
                        'mode': 'SPOT',
                        'target': best['pool_id'],
                        'risk': best['risk_profile']['overall_risk'],
                        'price': best['current_price']
                    }
        
        return {
            'action': 'HOLD',
            'mode': current_mode,
            'target': current_pool_id if current_mode == 'SPOT' else 'ON_DEMAND',
            'risk': current_risk if current_mode == 'SPOT' else 0,
            'price': current_price if current_mode == 'SPOT' else ondemand_price
        }

# ============================================================================
# BACKTESTER (Same as v7.3)
# ============================================================================

class Backtester:
    def __init__(self, decision_engine):
        self.decision_engine = decision_engine
        self.results = []
        self.costs = {
            'ml_spot': 0,
            'ml_ondemand': 0,
            'ml_total': 0,
            'single_pool_optimal': 0,
            'ondemand_only': 0
        }
        self.risk_tracking = {
            'ml': [],
            'single_pool': [],
            'ondemand': []
        }
    
    def run(self, df_test):
        print("\n" + "="*80)
        print("🧪 STEP 4: Backtesting with Validated Holiday Scores")
        print("="*80)
        
        instance_types = df_test['InstanceType'].unique()
        
        for instance_type in tqdm(instance_types, desc="Instance types"):
            inst_data = df_test[df_test['InstanceType'] == instance_type].copy()
            timestamps = inst_data['timestamp'].unique()[::config.DECISION_INTERVAL_HOURS]
            
            for current_time in timestamps:
                timestamp_data = inst_data[inst_data['timestamp'] == current_time]
                
                if len(timestamp_data) == 0:
                    continue
                
                decision = self.decision_engine.make_decision(instance_type, timestamp_data, current_time)
                
                next_time = current_time + timedelta(hours=config.DECISION_INTERVAL_HOURS)
                period_data = inst_data[
                    (inst_data['timestamp'] >= current_time) &
                    (inst_data['timestamp'] < next_time)
                ]
                
                if len(period_data) > 0:
                    # ML cost
                    if decision['mode'] == 'ONDEMAND':
                        cost = period_data['OnDemandPrice'].sum()
                        self.costs['ml_ondemand'] += cost
                        self.costs['ml_total'] += cost
                        self.risk_tracking['ml'].append(0)
                    else:
                        pool_data = period_data[period_data['Pool_ID'] == decision['target']]
                        if len(pool_data) > 0:
                            cost = pool_data['SpotPrice'].sum()
                            self.costs['ml_spot'] += cost
                            self.costs['ml_total'] += cost
                            self.risk_tracking['ml'].append(decision.get('risk', 0))
                    
                    # On-Demand only
                    self.costs['ondemand_only'] += period_data['OnDemandPrice'].sum()
                    self.risk_tracking['ondemand'].append(0)
                    
                    # Single pool with concentration penalty
                    for ts in period_data['timestamp'].unique():
                        ts_data = period_data[period_data['timestamp'] == ts]
                        if len(ts_data) > 0:
                            cheapest_price = ts_data['SpotPrice'].min()
                            cheapest_pool = ts_data.loc[ts_data['SpotPrice'].idxmin(), 'Pool_ID']
                            self.costs['single_pool_optimal'] += cheapest_price
                            
                            cheapest_data = ts_data[ts_data['Pool_ID'] == cheapest_pool]
                            if len(cheapest_data) > 0:
                                risk_info = self.decision_engine.risk_scorer.compute_overall_risk(
                                    cheapest_pool, cheapest_data, ts, True,
                                    is_single_pool_strategy=True
                                )
                                self.risk_tracking['single_pool'].append(risk_info['overall_risk'])
                
                self.results.append({
                    'timestamp': current_time,
                    'instance_type': instance_type,
                    **decision
                })
        
        print(f"✅ Complete: {len(self.results):,} decisions")
    
    def calculate_metrics(self):
        print("\n" + "="*80)
        print("📊 STEP 5: RESULTS WITH VALIDATED HOLIDAY SCORES")
        print("="*80)
        
        ml_total = self.costs['ml_total']
        ml_spot = self.costs['ml_spot']
        ml_ondemand = self.costs['ml_ondemand']
        single_optimal = self.costs['single_pool_optimal']
        ondemand = self.costs['ondemand_only']
        
        savings_vs_optimal = ((single_optimal - ml_total) / single_optimal * 100) if single_optimal > 0 else 0
        savings_vs_ondemand = ((ondemand - ml_total) / ondemand * 100) if ondemand > 0 else 0
        spot_pct = (ml_spot / ml_total * 100) if ml_total > 0 else 0
        
        switches = sum(1 for r in self.results if 'SWITCH' in r['action'])
        ondemand_periods = sum(1 for r in self.results if r['mode'] == 'ONDEMAND')
        
        ml_avg_risk = np.mean(self.risk_tracking['ml']) if self.risk_tracking['ml'] else 0
        single_avg_risk = np.mean(self.risk_tracking['single_pool']) if self.risk_tracking['single_pool'] else 0
        
        annual_ml = ml_total * 4
        annual_single = single_optimal * 4
        annual_ondemand = ondemand * 4
        
        metrics = {
            'total_decisions': len(self.results),
            'total_switches': switches,
            'ondemand_periods': ondemand_periods,
            'q1_ml_total_cost': ml_total,
            'q1_ml_spot_cost': ml_spot,
            'q1_ml_ondemand_cost': ml_ondemand,
            'q1_single_pool_optimal_cost': single_optimal,
            'q1_ondemand_only_cost': ondemand,
            'q1_savings_vs_optimal_pct': savings_vs_optimal,
            'q1_savings_vs_ondemand_pct': savings_vs_ondemand,
            'spot_usage_pct': spot_pct,
            'ml_avg_risk': ml_avg_risk,
            'single_pool_avg_risk': single_avg_risk,
            'annual_ml_cost': annual_ml,
            'annual_single_pool_cost': annual_single,
            'annual_ondemand_cost': annual_ondemand
        }
        
        print(f"\n💰 Q1 Costs:")
        print(f"   ML Strategy: ${ml_total:.2f}")
        print(f"   ├─ Spot: ${ml_spot:.2f} ({spot_pct:.1f}%)")
        print(f"   └─ On-Demand: ${ml_ondemand:.2f} ({100-spot_pct:.1f}%)")
        print(f"   Single Pool: ${single_optimal:.2f}")
        print(f"   On-Demand Only: ${ondemand:.2f}")
        
        print(f"\n💡 Q1 Savings:")
        print(f"   vs Single Pool: {savings_vs_optimal:+.2f}% (${single_optimal - ml_total:+.2f})")
        print(f"   vs On-Demand: {savings_vs_ondemand:+.1f}%")
        
        print(f"\n⚠️  Average Risk:")
        print(f"   ML Strategy: {ml_avg_risk:.1f}/100")
        print(f"   Single Pool: {single_avg_risk:.1f}/100")
        print(f"   Risk Reduction: {single_avg_risk - ml_avg_risk:.1f} points")
        
        print(f"\n📊 Decisions:")
        print(f"   Total: {len(self.results):,}")
        print(f"   Switches: {switches}")
        print(f"   On-Demand Periods: {ondemand_periods} ({ondemand_periods/len(self.results)*100:.1f}%)")
        
        print(f"\n📅 Annual Projection:")
        print(f"   ML: ${annual_ml:.2f}")
        print(f"   Single Pool: ${annual_single:.2f}")
        print(f"   Savings: ${annual_single - annual_ml:+.2f}")
        
        return metrics

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "🚀"*40)
    print("STARTING v7.3 WITH HOLIDAY VALIDATOR")
    print("🚀"*40)
    
    # Load data
    df_train = load_and_clean(config.TRAINING_DATA, "Training Data (2023-2024)")
    df_test = load_and_clean(config.TEST_DATA, "Test Data (Q1 2025)")
    df_test = df_test[df_test['InstanceType'].isin(config.INSTANCE_TYPES)].copy()
    
    # STEP 1: Holiday Validator
    holiday_validator = HolidayValidator()
    
    if not holiday_validator.load():
        print("\n⚙️  Running Holiday Validation (first time)...")
        holiday_validator.validate_holidays(df_train, config.EVENT_DATA)
        holiday_validator.save()
    
    # STEP 2: Event Calendar with validated scores
    event_calendar = EventCalendar(config.EVENT_DATA, holiday_validator)
    
    # STEP 3: Risk Scorer
    risk_scorer = RiskScorer(event_calendar)
    
    # STEP 4: Prophet Forecaster
    forecaster = ProphetForecaster(event_calendar)
    
    if not forecaster.load_models():
        print("\n⚙️  Training Prophet models...")
        forecaster.train_all(df_train)
        forecaster.save_models()
    
    # STEP 5: Decision Engine
    decision_engine = DecisionEngine(forecaster, risk_scorer)
    
    # STEP 6: Backtest
    backtester = Backtester(decision_engine)
    backtester.run(df_test)
    metrics = backtester.calculate_metrics()
    
    # Save results
    print("\n💾 Saving Results...")
    if backtester.results:
        results_df = pd.DataFrame(backtester.results)
        results_df.to_csv(os.path.join(config.OUTPUT_DIR, 'backtest_results.csv'), index=False)
        print(f"   ✅ Results saved")
    
    if metrics:
        with open(os.path.join(config.OUTPUT_DIR, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"   ✅ Metrics saved")
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"\n🎉 SUMMARY:")
    print(f"   Q1 ML Cost: ${metrics['q1_ml_total_cost']:.2f}")
    print(f"   Q1 Single Pool: ${metrics['q1_single_pool_optimal_cost']:.2f}")
    print(f"   Savings: {metrics['q1_savings_vs_optimal_pct']:+.2f}%")
    print(f"   Risk: {metrics['ml_avg_risk']:.1f}/100 (vs {metrics['single_pool_avg_risk']:.1f}/100)")
    print(f"   Annual Insurance: ${(metrics['q1_ml_total_cost'] - metrics['q1_single_pool_optimal_cost'])*4:.2f}")
    
    return metrics

if __name__ == "__main__":
    results = main()
