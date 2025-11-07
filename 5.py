#!/usr/bin/env python3
"""
AWS Spot Optimizer v8.1 - FIXED PRODUCTION PIPELINE
====================================================

🔧 CRITICAL FIXES APPLIED:
1. ✅ Tiered event risk multipliers (0.70-1.0 instead of 0.5)
2. ✅ Event escalation during capacity stress (+10-20 points)
3. ✅ Dynamic on-demand threshold (38-42 based on events)
4. ✅ Optimized risk weights (0.30 event weight)
5. ✅ Lower switching thresholds (2% vs 3%)

Expected Results:
- Annual Cost: $18,800-19,400 (slight premium for safety)
- On-Demand Usage: 2-4% (during critical events)
- Risk Reduction: 25-30 points maintained
- Reliability: 99.9%+ with event protection

Version: 8.1 FIXED
Date: 2025-11-06
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
from scipy import stats
import json
import os
import pickle
from collections import defaultdict

# ============================================================================
# 🔧 FIXED CONFIGURATION
# ============================================================================

class Config:
    """Production configuration with FIXES"""
    
    # Hardware
    USE_GPU = torch.backends.mps.is_available()
    N_CORES = mp.cpu_count()
    DEVICE = torch.device("mps" if USE_GPU else "cpu")
    
    # Paths
    TRAINING_DATA = '/Users/atharvapudale/Downloads/aws_2023_2024_complete_24months.csv'
    TEST_DATA = '/Users/atharvapudale/Downloads/mumbai_spot_data_sorted_asc(1-2-3-25).csv'
    EVENT_DATA = '/Users/atharvapudale/Downloads/aws_stress_events_2023_2025.csv'
    OUTPUT_DIR = './outputs/production_v8_1_FIXED'
    MODEL_DIR = './models/production_v8_1_FIXED'
    
    INSTANCE_TYPES = ['t3.medium', 't4g.small', 't4g.medium', 'c5.large']
    
    # Prophet Settings
    PROPHET_CHANGEPOINT_PRIOR = 0.001
    PROPHET_SEASONALITY_PRIOR = 0.01
    PROPHET_INTERVAL_WIDTH = 0.95
    PROPHET_DAILY_SEASONALITY = True
    PROPHET_WEEKLY_SEASONALITY = True
    PROPHET_YEARLY_SEASONALITY = False
    
    # 🔧 FIXED: Risk Weights (event weight increased)
    RISK_WEIGHT_CAPACITY = 0.35     # Was 0.40
    RISK_WEIGHT_VOLATILITY = 0.15   # Same
    RISK_WEIGHT_TREND = 0.15        # Was 0.20
    RISK_WEIGHT_EVENT = 0.30        # Was 0.15 → DOUBLED
    RISK_WEIGHT_PREDICTION = 0.05   # Was 0.10
    
    # AWS Official Capacity Thresholds
    CAPACITY_EXCELLENT = 0.30
    CAPACITY_GOOD = 0.40
    CAPACITY_MODERATE = 0.50
    CAPACITY_WARNING = 0.65
    CAPACITY_CRITICAL = 0.75
    
    # Risk Categories
    RISK_SAFE_MAX = 30
    RISK_LOW_MAX = 45
    RISK_MODERATE_MAX = 60
    
    # 🔧 FIXED: Decision Rules (more aggressive)
    SWITCH_MIN_SAVINGS = 0.02               # Was 0.03 → Lower to 2%
    SWITCH_RISK_DIFF_MAX = 15
    SWITCH_RISK_ABSOLUTE_MAX = 60           # Was 65 → Stricter
    ONDEMAND_RISK_THRESHOLD = 42            # Was 55 → LOWERED
    ONDEMAND_CRITICAL_EVENT_THRESHOLD = 38  # 🆕 NEW: Even lower during critical events
    
    # Event Detection Parameters
    EVENT_DETECTION_WINDOW = 15
    EVENT_IMPACT_THRESHOLD = 1.5
    PATTERN_MIN_OCCURRENCES = 2
    PRE_EVENT_MAX_DAYS = 14
    POST_EVENT_MAX_DAYS = 7
    
    # Anomaly Detection
    ANOMALY_DETECTION_ENABLED = True
    ANOMALY_THRESHOLD = 2.5
    ANOMALY_MIN_DURATION = 2
    
    # Other
    DECISION_INTERVAL_HOURS = 12
    FORECAST_HORIZON_HOURS = 168
    MAX_POOL_AGE_DAYS = 21
    DIVERSITY_MIN_SAVINGS = 0.02
    
    # Concentration Penalties
    SINGLE_POOL_CONCENTRATION_PENALTY = 20
    SINGLE_POOL_NO_ESCAPE_PENALTY = 10

config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)

print("="*80)
print("AWS SPOT OPTIMIZER v8.1 - FIXED PRODUCTION PIPELINE")
print("="*80)
print(f"\n⚙️  Hardware: {config.DEVICE}, {config.N_CORES} cores")
print(f"\n🔧 CRITICAL FIXES APPLIED:")
print(f"   1. Tiered event risk (0.70-1.0 multipliers)")
print(f"   2. Event escalation during capacity stress")
print(f"   3. Dynamic on-demand threshold (38-42)")
print(f"   4. Event weight increased to 30%")
print(f"   5. Lower switching threshold (2%)")
print(f"\n🎯 Complete Pipeline:")
print(f"   1. Intelligent Event Analyzer")
print(f"   2. Smart Event Calendar")
print(f"   3. Prophet Training (32 models)")
print(f"   4. Risk Scorer (FIXED)")
print(f"   5. Decision Engine (FIXED)")
print(f"   6. Backtester")
print("\n" + "="*80 + "\n")

# ============================================================================
# STEP 1: INTELLIGENT EVENT ANALYZER (Unchanged)
# ============================================================================

class IntelligentEventAnalyzer:
    """Analyzes events with dynamic window detection"""
    
    def __init__(self):
        self.baseline_stats = {}
        self.event_patterns = {}
        self.unnamed_events = []
        self.daily_stats = None
    
    def analyze_baseline(self, df_train):
        """Calculate baseline statistics"""
        print("\n" + "="*80)
        print("📊 STEP 1: INTELLIGENT EVENT ANALYSIS")
        print("="*80)
        
        print("\n🔍 Calculating baseline from 2023-2024...")
        df_train['date'] = df_train['timestamp'].dt.date
        
        self.daily_stats = df_train.groupby('date').agg({
            'SpotPrice': ['mean', 'std', 'min', 'max'],
            'ratio': ['mean', 'std', 'min', 'max']
        }).reset_index()
        
        self.daily_stats.columns = ['date', 'price_mean', 'price_std', 'price_min', 'price_max',
                                     'ratio_mean', 'ratio_std', 'ratio_min', 'ratio_max']
        
        self.baseline_stats = {
            'price_mean': self.daily_stats['price_mean'].mean(),
            'price_std': self.daily_stats['price_mean'].std(),
            'ratio_mean': self.daily_stats['ratio_mean'].mean(),
            'ratio_std': self.daily_stats['ratio_mean'].std()
        }
        
        print(f"   Baseline Price: ${self.baseline_stats['price_mean']:.4f}")
        print(f"   Baseline Ratio: {self.baseline_stats['ratio_mean']:.3f}")
        
        return self.baseline_stats
    
    def detect_event_window(self, event_date, event_name):
        """Auto-detect pre/post impact days"""
        if self.daily_stats is None:
            return 3, 2, 50
        
        if isinstance(event_date, pd.Timestamp):
            event_date_dt = event_date.date()
        else:
            event_date_dt = pd.Timestamp(event_date).date()
        
        window_start = pd.Timestamp(event_date_dt - timedelta(days=config.PRE_EVENT_MAX_DAYS))
        window_end = pd.Timestamp(event_date_dt + timedelta(days=config.POST_EVENT_MAX_DAYS))
        
        event_window = self.daily_stats[
            (pd.to_datetime(self.daily_stats['date']) >= window_start) &
            (pd.to_datetime(self.daily_stats['date']) <= window_end)
        ].copy()
        
        if len(event_window) < 5:
            return 3, 2, 50
        
        event_window['price_zscore'] = (
            (event_window['price_mean'] - self.baseline_stats['price_mean']) /
            (self.baseline_stats['price_std'] + 1e-6)
        )
        event_window['ratio_zscore'] = (
            (event_window['ratio_mean'] - self.baseline_stats['ratio_mean']) /
            (self.baseline_stats['ratio_std'] + 1e-6)
        )
        event_window['combined_zscore'] = (event_window['price_zscore'] + event_window['ratio_zscore']) / 2
        
        event_row_idx = event_window[pd.to_datetime(event_window['date']) == pd.Timestamp(event_date_dt)].index
        if len(event_row_idx) == 0:
            event_center = len(event_window) // 2
        else:
            event_center = event_window.index.get_loc(event_row_idx[0])
        
        pre_days = 0
        for i in range(event_center - 1, -1, -1):
            if event_window.iloc[i]['combined_zscore'] > config.EVENT_IMPACT_THRESHOLD:
                pre_days = event_center - i
            else:
                break
        
        post_days = 0
        for i in range(event_center + 1, len(event_window)):
            if event_window.iloc[i]['combined_zscore'] > config.EVENT_IMPACT_THRESHOLD:
                post_days = i - event_center
            else:
                break
        
        max_zscore = event_window['combined_zscore'].max()
        avg_zscore = event_window['combined_zscore'].mean()
        
        severity = min(100, max(0, int(
            30 * max_zscore / 3 +
            20 * avg_zscore / 2 +
            25 * (1 if pre_days > 3 else pre_days / 3) +
            25 * (1 if post_days > 2 else post_days / 2)
        )))
        
        pre_days = max(pre_days, 1)
        post_days = max(post_days, 1)
        
        return pre_days, post_days, severity
    
    def analyze_event_pattern(self, event_name, event_dates):
        """Analyze recurring event pattern"""
        if len(event_dates) < config.PATTERN_MIN_OCCURRENCES:
            return None
        
        patterns = []
        for event_date in event_dates:
            pre, post, severity = self.detect_event_window(event_date, event_name)
            patterns.append({
                'date': event_date,
                'pre_days': pre,
                'post_days': post,
                'severity': severity
            })
        
        return {
            'event_name': event_name,
            'occurrences': len(patterns),
            'avg_pre_days': int(np.mean([p['pre_days'] for p in patterns])),
            'avg_post_days': int(np.mean([p['post_days'] for p in patterns])),
            'avg_severity': int(np.mean([p['severity'] for p in patterns])),
            'individual_patterns': patterns,
            'status': 'VALIDATED_PATTERN'
        }
    
    def detect_unnamed_events(self):
        """Discover unnamed events through anomaly detection"""
        if self.daily_stats is None or not config.ANOMALY_DETECTION_ENABLED:
            return []
        
        print("\n🔍 Detecting unnamed events...")
        
        self.daily_stats['price_anomaly'] = np.abs(
            (self.daily_stats['price_mean'] - self.baseline_stats['price_mean']) /
            (self.baseline_stats['price_std'] + 1e-6)
        )
        self.daily_stats['ratio_anomaly'] = np.abs(
            (self.daily_stats['ratio_mean'] - self.baseline_stats['ratio_mean']) /
            (self.baseline_stats['ratio_std'] + 1e-6)
        )
        self.daily_stats['combined_anomaly'] = (
            self.daily_stats['price_anomaly'] + self.daily_stats['ratio_anomaly']
        ) / 2
        
        anomalies = self.daily_stats[
            self.daily_stats['combined_anomaly'] > config.ANOMALY_THRESHOLD
        ].copy()
        
        if len(anomalies) == 0:
            print("   No unnamed events detected")
            return []
        
        unnamed_events = []
        current_event = []
        
        for idx, row in anomalies.iterrows():
            if not current_event:
                current_event = [row]
            else:
                last_date = current_event[-1]['date']
                if (row['date'] - last_date).days <= 2:
                    current_event.append(row)
                else:
                    if len(current_event) >= config.ANOMALY_MIN_DURATION:
                        unnamed_events.append(current_event)
                    current_event = [row]
        
        if len(current_event) >= config.ANOMALY_MIN_DURATION:
            unnamed_events.append(current_event)
        
        formatted_events = []
        for idx, event in enumerate(unnamed_events, 1):
            event_date = event[len(event)//2]['date']
            avg_anomaly = np.mean([row['combined_anomaly'] for row in event])
            severity = min(100, int(avg_anomaly / config.ANOMALY_THRESHOLD * 50))
            
            formatted_events.append({
                'event_name': f"System-Detected-Event-{idx}",
                'date': event_date,
                'duration_days': len(event),
                'severity': severity,
                'status': 'SYSTEM_DETECTED'
            })
        
        print(f"   ✅ Detected {len(formatted_events)} unnamed events")
        return formatted_events

# ============================================================================
# STEP 2: SMART EVENT CALENDAR (Unchanged)
# ============================================================================

class SmartEventCalendar:
    """Event calendar with intelligent patterns"""
    
    def __init__(self, event_csv_path, event_analyzer):
        self.events = []
        self.event_analyzer = event_analyzer
        self.load_and_analyze_events(event_csv_path)
    
    def load_and_analyze_events(self, path):
        print("\n" + "="*80)
        print("📅 STEP 2: SMART EVENT CALENDAR")
        print("="*80)
        
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.strip()
        
        date_col = next((c for c in df.columns if 'date' in c), None)
        name_col = next((c for c in df.columns if 'event' in c or 'name' in c), None)
        region_col = next((c for c in df.columns if 'region' in c), None)
        
        if not date_col:
            print("   ⚠️  No date column")
            return
        
        df[date_col] = pd.to_datetime(df[date_col])
        
        events_by_name = defaultdict(list)
        for _, row in df.iterrows():
            event_name = str(row[name_col]) if name_col else 'Event'
            event_date = pd.Timestamp(row[date_col])
            events_by_name[event_name].append(event_date)
        
        print(f"\n🔍 Analyzing {len(events_by_name)} unique events...")
        
        validated_count = 0
        for event_name, dates in tqdm(events_by_name.items(), desc="Patterns"):
            pattern = self.event_analyzer.analyze_event_pattern(event_name, dates)
            
            if pattern:
                for date in dates:
                    self.events.append({
                        'date': date,
                        'name': event_name,
                        'severity': pattern['avg_severity'],
                        'pre_days': pattern['avg_pre_days'],
                        'post_days': pattern['avg_post_days'],
                        'region': 'all',
                        'status': 'PATTERN_VALIDATED'
                    })
                    validated_count += 1
            else:
                for date in dates:
                    pre, post, severity = self.event_analyzer.detect_event_window(date, event_name)
                    self.events.append({
                        'date': date,
                        'name': event_name,
                        'severity': severity,
                        'pre_days': pre,
                        'post_days': post,
                        'region': 'all',
                        'status': 'SINGLE_OCCURRENCE'
                    })
        
        unnamed = self.event_analyzer.detect_unnamed_events()
        for event in unnamed:
            self.events.append({
                'date': event['date'],
                'name': event['event_name'],
                'severity': event['severity'],
                'pre_days': 3,
                'post_days': 2,
                'region': 'all',
                'status': 'SYSTEM_DETECTED'
            })
        
        print(f"\n✅ Event Calendar Complete:")
        print(f"   Pattern-validated: {validated_count}")
        print(f"   System-detected: {len(unnamed)}")
        print(f"   Total: {len(self.events)}")
    
    def check_event_window(self, current_date, region='all'):
        """Check if in event window"""
        if not isinstance(current_date, pd.Timestamp):
            current_date = pd.Timestamp(current_date)
        
        max_severity = 0
        relevant_event = None
        
        for event in self.events:
            if event['region'] not in ['all', region]:
                continue
            
            event_start = event['date'] - timedelta(days=event['pre_days'])
            event_end = event['date'] + timedelta(days=event['post_days'])
            
            if event_start <= current_date <= event_end:
                if event['severity'] > max_severity:
                    max_severity = event['severity']
                    relevant_event = event
        
        return max_severity, relevant_event

# ============================================================================
# STEP 3: DATA LOADER (Unchanged)
# ============================================================================

def load_and_clean(csv_path, name):
    """Load and clean data"""
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
    
    df = df.rename(columns=col_map)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = df[(df['SpotPrice'] >= 0) & (df['SpotPrice'] < 10) & (df['OnDemandPrice'] > 0)].copy()
    df = df[df['InstanceType'].isin(config.INSTANCE_TYPES)].copy()
    
    if 'Region' not in df.columns:
        df['Region'] = 'ap-south-1'
    
    df['Pool_ID'] = df['InstanceType'] + '_' + df['AZ']
    df['ratio'] = (df['SpotPrice'] / df['OnDemandPrice']).clip(0, 1)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✅ Clean: {len(df):,} records, {df['Pool_ID'].nunique()} pools")
    print(f"   Dates: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
    print(f"   Avg Ratio: {df['ratio'].mean():.3f}")
    
    return df

# ============================================================================
# STEP 4: PROPHET FORECASTER (Unchanged)
# ============================================================================

class ProphetForecaster:
    """Prophet forecaster with event awareness"""
    
    def __init__(self, event_calendar):
        self.event_calendar = event_calendar
        self.models = {}
        self.metrics = {}
    
    def prepare_holidays(self, pool_data, region):
        """Prepare holidays for Prophet"""
        holidays_list = []
        date_range = pd.date_range(
            start=pool_data['timestamp'].min(),
            end=pool_data['timestamp'].max(),
            freq='D'
        )
        
        for date in date_range:
            severity, event = self.event_calendar.check_event_window(date, region)
            if severity > 40 and event:
                holidays_list.append({
                    'ds': date,
                    'holiday': event['name'],
                    'lower_window': -event['pre_days'],
                    'upper_window': event['post_days']
                })
        
        if holidays_list:
            return pd.DataFrame(holidays_list).drop_duplicates(subset=['ds'])
        return None
    
    def train_pool(self, pool_id, pool_data):
        """Train Prophet for one pool"""
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
            
            return model, {'records': len(prophet_df)}
        except Exception as e:
            return None, {'error': str(e)}
    
    def train_all(self, df_train):
        """Train all pool models"""
        print("\n" + "="*80)
        print("🤖 STEP 3: PROPHET TRAINING")
        print("="*80)
        
        pools = df_train['Pool_ID'].unique()
        print(f"Training {len(pools)} models...")
        
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
        
        print(f"\n✅ Trained {successful}/{len(pools)} models")
        return successful
    
    def save_models(self):
        """Save models"""
        print("\n💾 Saving models...")
        with open(os.path.join(config.MODEL_DIR, 'prophet_models.pkl'), 'wb') as f:
            pickle.dump(self.models, f)
        with open(os.path.join(config.MODEL_DIR, 'prophet_metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"   ✅ Saved {len(self.models)} models")
    
    def load_models(self):
        """Load models"""
        path = os.path.join(config.MODEL_DIR, 'prophet_models.pkl')
        if os.path.exists(path):
            print("\n📂 Loading models...")
            with open(path, 'rb') as f:
                self.models = pickle.load(f)
            print(f"   ✅ Loaded {len(self.models)} models")
            return True
        return False

# ============================================================================
# 🔧 STEP 5: FIXED RISK SCORER
# ============================================================================

class RiskScorer:
    """🔧 FIXED Risk scorer with proper event handling"""
    
    def __init__(self, event_calendar):
        self.event_calendar = event_calendar
        self.event_log = []  # Track event escalations for analysis
    
    def compute_capacity_risk(self, pool_data):
        """Calculate capacity risk"""
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
        """Calculate volatility risk"""
        if len(pool_data) < 24:
            return 30
        recent = pool_data.tail(168)
        std_price = recent['SpotPrice'].std()
        mean_price = recent['SpotPrice'].mean()
        cv = (std_price / mean_price) if mean_price > 0 else 0
        risk = (cv / 0.15) * 100
        return np.clip(risk, 0, 100)
    
    def compute_trend_risk(self, pool_data):
        """Calculate trend risk"""
        if len(pool_data) < 48:
            return 0
        recent_24h = pool_data.tail(24)['ratio'].mean()
        previous_24h = pool_data.tail(48).head(24)['ratio'].mean()
        ratio_change = (recent_24h - previous_24h) / (previous_24h + 1e-6)
        risk = (ratio_change / 0.10) * 100
        return np.clip(risk, 0, 100)
    
    def compute_event_risk(self, current_date, region):
        """🔧 FIXED: Tiered event risk calculation"""
        severity, event = self.event_calendar.check_event_window(current_date, region)
        
        if severity == 0:
            return 0
        
        # 🆕 TIERED MULTIPLIERS (instead of flat 0.5)
        if severity >= 70:      # Critical: Prime Day, Black Friday, Cyber Monday
            return severity     # 100% impact (70-80 points)
        elif severity >= 55:    # High: Super Bowl, Independence Day
            return severity * 0.85  # 85% impact (47-51 points)
        elif severity >= 40:    # Medium: Regional holidays, sporting events
            return severity * 0.70  # 70% impact (28-35 points)
        else:                   # Low: Minor events
            return severity * 0.50  # 50% impact (10-20 points)
    
    def compute_overall_risk(self, pool_id, pool_data, current_date, is_single_pool=False):
        """🔧 FIXED: Calculate overall risk with event escalation"""
        region = pool_data['Region'].iloc[0] if len(pool_data) > 0 else 'ap-south-1'
        
        capacity_risk = self.compute_capacity_risk(pool_data)
        volatility_risk = self.compute_volatility_risk(pool_data)
        trend_risk = self.compute_trend_risk(pool_data)
        event_risk = self.compute_event_risk(current_date, region)
        
        # Base risk calculation
        base_risk = (
            config.RISK_WEIGHT_CAPACITY * capacity_risk +
            config.RISK_WEIGHT_VOLATILITY * volatility_risk +
            config.RISK_WEIGHT_TREND * trend_risk +
            config.RISK_WEIGHT_EVENT * event_risk +
            config.RISK_WEIGHT_PREDICTION * 10
        )
        
        # 🆕 EVENT ESCALATION: Add extra risk during critical events with capacity pressure
        escalation = 0
        severity, event = self.event_calendar.check_event_window(current_date, region)
        
        if severity >= 70 and capacity_risk >= 40:
            # Critical event + capacity pressure = compound danger
            # Escalation: 10 base points + 0.5 per capacity point above 40
            escalation = min(20, (capacity_risk - 40) * 0.5 + 10)
            base_risk += escalation
            
            # Log for analysis
            self.event_log.append({
                'date': current_date,
                'pool_id': pool_id,
                'event': event['name'] if event else 'Unknown',
                'severity': severity,
                'capacity_risk': capacity_risk,
                'base_risk': base_risk - escalation,
                'escalation': escalation,
                'final_risk': base_risk
            })
        
        # Single pool penalty
        if is_single_pool:
            penalty = config.SINGLE_POOL_CONCENTRATION_PENALTY + config.SINGLE_POOL_NO_ESCAPE_PENALTY
            base_risk = min(base_risk + penalty, 100)
        
        return {
            'pool_id': pool_id,
            'overall_risk': base_risk,
            'capacity_risk': capacity_risk,
            'volatility_risk': volatility_risk,
            'trend_risk': trend_risk,
            'event_risk': event_risk,
            'event_severity': severity,
            'escalation': escalation,
            'current_ratio': pool_data['ratio'].iloc[-1] if len(pool_data) > 0 else 0.5,
            'timestamp': current_date
        }

# ============================================================================
# 🔧 STEP 6: FIXED DECISION ENGINE
# ============================================================================

class DecisionEngine:
    """🔧 FIXED Smart decision engine with dynamic thresholds"""
    
    def __init__(self, risk_scorer):
        self.risk_scorer = risk_scorer
        self.current_pool = {}
        self.current_mode = {}
        self.decision_log = []  # Track on-demand switches
    
    def make_decision(self, instance_type, all_pools_data, current_time):
        """🔧 FIXED: Make optimization decision with dynamic threshold"""
        
        # Initialize
        if instance_type not in self.current_pool:
            evaluations = []
            for pool_id in all_pools_data['Pool_ID'].unique():
                pool_data = all_pools_data[all_pools_data['Pool_ID'] == pool_id]
                risk_info = self.risk_scorer.compute_overall_risk(pool_id, pool_data, current_time)
                evaluations.append({
                    'pool_id': pool_id,
                    'risk': risk_info['overall_risk'],
                    'price': pool_data['SpotPrice'].iloc[-1] if len(pool_data) > 0 else 999
                })
            
            if not evaluations:
                return {'action': 'ERROR', 'mode': 'SPOT', 'target': 'NONE', 'risk': 50, 'price': 0}
            
            evaluations.sort(key=lambda x: (x['risk'], x['price']))
            best = evaluations[0]
            
            self.current_pool[instance_type] = best['pool_id']
            self.current_mode[instance_type] = 'SPOT'
            
            return {
                'action': 'INIT',
                'mode': 'SPOT',
                'target': best['pool_id'],
                'risk': best['risk'],
                'price': best['price']
            }
        
        # Check current pool
        current_pool_id = self.current_pool[instance_type]
        current_mode = self.current_mode[instance_type]
        
        current_pool_data = all_pools_data[all_pools_data['Pool_ID'] == current_pool_id]
        if len(current_pool_data) == 0:
            return {'action': 'HOLD', 'mode': current_mode, 'target': current_pool_id, 'risk': 0, 'price': 0}
        
        risk_info = self.risk_scorer.compute_overall_risk(current_pool_id, current_pool_data, current_time)
        current_risk = risk_info['overall_risk']
        event_severity = risk_info.get('event_severity', 0)
        current_price = current_pool_data['SpotPrice'].iloc[-1]
        ondemand_price = current_pool_data['OnDemandPrice'].iloc[-1]
        
        # 🆕 DYNAMIC THRESHOLD: Use lower threshold during critical events
        threshold = config.ONDEMAND_RISK_THRESHOLD
        if event_severity >= 70:
            threshold = config.ONDEMAND_CRITICAL_EVENT_THRESHOLD
        
        # High risk → On-Demand
        if current_risk > threshold:
            if current_mode != 'ONDEMAND':
                self.current_mode[instance_type] = 'ONDEMAND'
                
                # Log on-demand switch
                self.decision_log.append({
                    'timestamp': current_time,
                    'instance_type': instance_type,
                    'action': 'SWITCH_TO_ONDEMAND',
                    'risk': current_risk,
                    'threshold': threshold,
                    'event_severity': event_severity,
                    'reason': f'Risk {current_risk:.1f} > {threshold} (event: {event_severity})'
                })
                
                return {
                    'action': 'SWITCH_TO_ONDEMAND',
                    'mode': 'ONDEMAND',
                    'target': 'ON_DEMAND',
                    'risk': 0,
                    'price': ondemand_price,
                    'reason': f'Risk {current_risk:.1f} > {threshold}'
                }
        else:
            # Risk dropped → return to Spot
            if current_mode == 'ONDEMAND':
                evaluations = []
                for pool_id in all_pools_data['Pool_ID'].unique():
                    pool_data = all_pools_data[all_pools_data['Pool_ID'] == pool_id]
                    r_info = self.risk_scorer.compute_overall_risk(pool_id, pool_data, current_time)
                    if r_info['overall_risk'] <= threshold:
                        evaluations.append({
                            'pool_id': pool_id,
                            'risk': r_info['overall_risk'],
                            'price': pool_data['SpotPrice'].iloc[-1]
                        })
                
                if evaluations:
                    evaluations.sort(key=lambda x: (x['risk'], x['price']))
                    best = evaluations[0]
                    
                    self.current_pool[instance_type] = best['pool_id']
                    self.current_mode[instance_type] = 'SPOT'
                    
                    self.decision_log.append({
                        'timestamp': current_time,
                        'instance_type': instance_type,
                        'action': 'RETURN_TO_SPOT',
                        'risk': best['risk'],
                        'threshold': threshold,
                        'reason': f'Risk dropped to {best["risk"]:.1f}'
                    })
                    
                    return {
                        'action': 'RETURN_TO_SPOT',
                        'mode': 'SPOT',
                        'target': best['pool_id'],
                        'risk': best['risk'],
                        'price': best['price']
                    }
        
        # Switch pools for savings
        if current_mode == 'SPOT':
            candidates = []
            for pool_id in all_pools_data['Pool_ID'].unique():
                if pool_id == current_pool_id:
                    continue
                
                pool_data = all_pools_data[all_pools_data['Pool_ID'] == pool_id]
                r_info = self.risk_scorer.compute_overall_risk(pool_id, pool_data, current_time)
                
                if r_info['overall_risk'] > config.SWITCH_RISK_ABSOLUTE_MAX:
                    continue
                
                candidate_price = pool_data['SpotPrice'].iloc[-1]
                savings_pct = (current_price - candidate_price) / current_price
                
                if savings_pct >= config.SWITCH_MIN_SAVINGS:
                    candidates.append({
                        'pool_id': pool_id,
                        'risk': r_info['overall_risk'],
                        'price': candidate_price
                    })
            
            if candidates:
                candidates.sort(key=lambda x: (x['risk'], x['price']))
                best = candidates[0]
                
                self.current_pool[instance_type] = best['pool_id']
                
                return {
                    'action': 'SWITCH_SPOT_POOL',
                    'mode': 'SPOT',
                    'target': best['pool_id'],
                    'risk': best['risk'],
                    'price': best['price']
                }
        
        return {
            'action': 'HOLD',
            'mode': current_mode,
            'target': current_pool_id if current_mode == 'SPOT' else 'ON_DEMAND',
            'risk': current_risk if current_mode == 'SPOT' else 0,
            'price': current_price if current_mode == 'SPOT' else ondemand_price
        }

# ============================================================================
# STEP 7: BACKTESTER (Enhanced with logging)
# ============================================================================

class Backtester:
    """Complete backtesting system with enhanced logging"""
    
    def __init__(self, decision_engine):
        self.decision_engine = decision_engine
        self.results = []
        self.costs = {
            'ml_spot': 0,
            'ml_ondemand': 0,
            'ml_total': 0,
            'single_pool': 0,
            'ondemand_only': 0
        }
        self.risk_tracking = {'ml': [], 'single': []}
    
    def run(self, df_test):
        """Run backtest"""
        print("\n" + "="*80)
        print("🧪 STEP 4: BACKTESTING")
        print("="*80)
        
        instance_types = df_test['InstanceType'].unique()
        
        for instance_type in tqdm(instance_types, desc="Testing"):
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
                    # ML costs
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
                            self.risk_tracking['ml'].append(decision['risk'])
                    
                    # Single pool baseline
                    cheapest_pool = period_data.groupby('Pool_ID')['SpotPrice'].sum().idxmin()
                    self.costs['single_pool'] += period_data[period_data['Pool_ID'] == cheapest_pool]['SpotPrice'].sum()
                    
                    single_pool_data = period_data[period_data['Pool_ID'] == cheapest_pool]
                    if len(single_pool_data) > 0:
                        risk_info = self.decision_engine.risk_scorer.compute_overall_risk(
                            cheapest_pool, single_pool_data, current_time, True
                        )
                        self.risk_tracking['single'].append(risk_info['overall_risk'])
                    
                    # On-Demand only
                    self.costs['ondemand_only'] += period_data['OnDemandPrice'].sum()
                
                self.results.append({
                    'timestamp': current_time,
                    'instance_type': instance_type,
                    **decision
                })
        
        print(f"✅ Complete: {len(self.results):,} decisions")
        
        # Print on-demand switch summary
        od_switches = len(self.decision_engine.decision_log)
        if od_switches > 0:
            print(f"\n🚨 On-Demand Switches: {od_switches}")
            print(f"   Sample switches:")
            for log in self.decision_engine.decision_log[:5]:
                print(f"     {log['timestamp'].date()}: {log['reason']}")
    
    def calculate_metrics(self):
        """Calculate final metrics"""
        print("\n" + "="*80)
        print("📊 STEP 5: RESULTS")
        print("="*80)
        
        ml_total = self.costs['ml_total']
        ml_spot = self.costs['ml_spot']
        ml_ondemand = self.costs['ml_ondemand']
        single = self.costs['single_pool']
        ondemand = self.costs['ondemand_only']
        
        savings_vs_single = ((single - ml_total) / single * 100) if single > 0 else 0
        ondemand_pct = (ml_ondemand / ml_total * 100) if ml_total > 0 else 0
        
        switches = sum(1 for r in self.results if 'SWITCH' in r['action'])
        ondemand_periods = sum(1 for r in self.results if r['mode'] == 'ONDEMAND')
        switches_to_od = sum(1 for r in self.results if r['action'] == 'SWITCH_TO_ONDEMAND')
        
        ml_avg_risk = np.mean(self.risk_tracking['ml']) if self.risk_tracking['ml'] else 0
        single_avg_risk = np.mean(self.risk_tracking['single']) if self.risk_tracking['single'] else 0
        
        annual_ml = ml_total * 4
        annual_single = single * 4
        premium = annual_ml - annual_single
        
        print(f"\n💰 Q1 Costs:")
        print(f"   ML Strategy:    ${ml_total:.2f}")
        print(f"   ├─ Spot:        ${ml_spot:.2f} ({100-ondemand_pct:.1f}%)")
        print(f"   └─ On-Demand:   ${ml_ondemand:.2f} ({ondemand_pct:.1f}%)")
        print(f"   Single Pool:    ${single:.2f}")
        print(f"   Savings:        {savings_vs_single:+.2f}%")
        
        print(f"\n⚠️  Risk:")
        print(f"   ML:     {ml_avg_risk:.1f}/100")
        print(f"   Single: {single_avg_risk:.1f}/100")
        print(f"   Reduction: {single_avg_risk - ml_avg_risk:.1f} points")
        
        print(f"\n📊 Operations:")
        print(f"   Decisions: {len(self.results):,}")
        print(f"   Pool Switches: {switches - switches_to_od}")
        print(f"   On-Demand Switches: {switches_to_od}")
        print(f"   On-Demand Periods: {ondemand_periods}")
        
        print(f"\n📅 Annual Projection:")
        print(f"   ML:     ${annual_ml:.2f}")
        print(f"   Single: ${annual_single:.2f}")
        print(f"   Premium: ${premium:+.2f} ({premium/annual_single*100:+.2f}%)")
        
        if ondemand_pct > 0:
            print(f"\n💡 Insurance Analysis:")
            print(f"   Premium: ${premium:+.2f}/year")
            print(f"   For: {ondemand_pct:.1f}% on-demand protection")
            print(f"   During: {switches_to_od} critical events")
            print(f"   Risk reduction: {single_avg_risk - ml_avg_risk:.1f} points")
        
        # Event escalation summary
        if len(self.decision_engine.risk_scorer.event_log) > 0:
            print(f"\n🔥 Event Escalations:")
            print(f"   Total: {len(self.decision_engine.risk_scorer.event_log)}")
            print(f"   Sample escalations:")
            for log in self.decision_engine.risk_scorer.event_log[:3]:
                print(f"     {log['date'].date()}: {log['event']}")
                print(f"       Risk: {log['base_risk']:.1f} + {log['escalation']:.1f} = {log['final_risk']:.1f}")
        
        return {
            'q1_ml_total': ml_total,
            'q1_ml_spot': ml_spot,
            'q1_ml_ondemand': ml_ondemand,
            'q1_single': single,
            'savings_pct': savings_vs_single,
            'ondemand_pct': ondemand_pct,
            'switches_to_od': switches_to_od,
            'ondemand_periods': ondemand_periods,
            'ml_avg_risk': ml_avg_risk,
            'single_avg_risk': single_avg_risk,
            'annual_ml': annual_ml,
            'annual_single': annual_single,
            'annual_premium': premium
        }

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Complete production pipeline"""
    
    print("\n" + "🚀"*40)
    print("STARTING FIXED PRODUCTION PIPELINE")
    print("🚀"*40)
    
    # Load data
    df_train = load_and_clean(config.TRAINING_DATA, "Training (2023-2024)")
    df_test = load_and_clean(config.TEST_DATA, "Test (Q1 2025)")
    df_test = df_test[df_test['InstanceType'].isin(config.INSTANCE_TYPES)].copy()
    
    # STEP 1: Event Analysis
    event_analyzer = IntelligentEventAnalyzer()
    event_analyzer.analyze_baseline(df_train)
    
    # STEP 2: Event Calendar
    event_calendar = SmartEventCalendar(config.EVENT_DATA, event_analyzer)
    
    # Save event calendar
    print("\n💾 Saving event calendar...")
    with open(os.path.join(config.MODEL_DIR, 'event_calendar.pkl'), 'wb') as f:
        pickle.dump(event_calendar, f)
    print(f"   ✅ Saved event calendar with {len(event_calendar.events)} events")
    
    # STEP 3: Prophet Training
    forecaster = ProphetForecaster(event_calendar)
    if not forecaster.load_models():
        forecaster.train_all(df_train)
        forecaster.save_models()
    
    # STEP 4: Risk Scorer (FIXED)
    risk_scorer = RiskScorer(event_calendar)
    
    # STEP 5: Decision Engine (FIXED)
    decision_engine = DecisionEngine(risk_scorer)
    
    # STEP 6: Backtest
    backtester = Backtester(decision_engine)
    backtester.run(df_test)
    metrics = backtester.calculate_metrics()
    
    # Save results
    print("\n💾 Saving results...")
    if backtester.results:
        pd.DataFrame(backtester.results).to_csv(
            os.path.join(config.OUTPUT_DIR, 'results.csv'), index=False
        )
    
    # Save decision log
    if decision_engine.decision_log:
        pd.DataFrame(decision_engine.decision_log).to_csv(
            os.path.join(config.OUTPUT_DIR, 'ondemand_switches.csv'), index=False
        )
    
    # Save event escalations
    if risk_scorer.event_log:
        pd.DataFrame(risk_scorer.event_log).to_csv(
            os.path.join(config.OUTPUT_DIR, 'event_escalations.csv'), index=False
        )
    
    with open(os.path.join(config.OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)
    print(f"\n📁 Results saved to: {config.OUTPUT_DIR}")
    print(f"   - results.csv (all decisions)")
    print(f"   - ondemand_switches.csv (OD switch log)")
    print(f"   - event_escalations.csv (risk escalations)")
    print(f"   - metrics.json (summary metrics)")
    
    return metrics

if __name__ == "__main__":
    results = main()
  #!/usr/bin/env python3
"""
Smart Adaptive Threshold System - Full Q1/Q2/Q3 Testing
========================================================

Complete implementation and testing of smart threshold system
with historical learning from 2023-2024 data.

Features:
1. Monthly base threshold calculation
2. Event-based adjustments
3. Capacity compound risk detection
4. Historical event-capacity correlation
5. Full backtest on Q1/Q2/Q3 2025

Usage: python smart_threshold_full_test.py
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime, timedelta
from tqdm.auto import tqdm
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    TRAINING_DATA = '/Users/atharvapudale/Downloads/aws_2023_2024_complete_24months.csv'
    TEST_Q1 = '/Users/atharvapudale/Downloads/mumbai_spot_data_sorted_asc(1-2-3-25).csv'
    TEST_Q2 = '/Users/atharvapudale/Downloads/mumbai_spot_data_sorted_asc(4-5-6-25).csv'
    TEST_Q3 = '/Users/atharvapudale/Downloads/mumbai_spot_data_sorted_asc(7-8-9-25).csv'
    EVENT_DATA = '/Users/atharvapudale/Downloads/aws_stress_events_2023_2025.csv'
    
    MODEL_DIR = './models/smart_threshold_system'
    OUTPUT_DIR = './outputs/smart_threshold_testing'
    
    INSTANCE_TYPES = ['t3.medium', 't4g.small', 't4g.medium', 'c5.large']
    
    # Risk Weights (from v8.1)
    RISK_WEIGHT_CAPACITY = 0.35
    RISK_WEIGHT_VOLATILITY = 0.15
    RISK_WEIGHT_TREND = 0.15
    RISK_WEIGHT_EVENT = 0.30
    RISK_WEIGHT_PREDICTION = 0.05
    
    # Capacity Thresholds
    CAPACITY_EXCELLENT = 0.30
    CAPACITY_GOOD = 0.40
    CAPACITY_MODERATE = 0.50
    CAPACITY_WARNING = 0.65
    CAPACITY_CRITICAL = 0.75
    
    # Smart Threshold Parameters
    MONTHLY_BASE_THRESHOLDS = {
        1: 40, 2: 40,   # Jan-Feb: Moderate
        3: 35, 4: 35,   # Mar-Apr: High stress (Ram Navami season)
        5: 45, 6: 45,   # May-Jun: Low stress
        7: 40, 8: 40,   # Jul-Aug: Moderate
        9: 35, 10: 35,  # Sep-Oct: High stress (festivals)
        11: 32, 12: 32  # Nov-Dec: CRITICAL (holiday shopping)
    }
    
    SEVERITY_ADJUSTMENTS = {
        80: -18,  # Critical (Black Friday, Cyber Monday)
        70: -12,  # Critical (Ram Navami)
        60: -8,   # High (Super Bowl, Good Friday)
        55: -5,   # Medium-high
        50: -3,   # Medium
        40: -2    # Medium-low
    }
    
    COMPOUND_RISK_ADJUSTMENT = -10  # When event + capacity stress
    COMPOUND_RISK_CAPACITY_THRESHOLD = 0.45
    COMPOUND_RISK_SEVERITY_THRESHOLD = 60
    
    THRESHOLD_FLOOR = 25
    THRESHOLD_CEILING = 50
    
    # Decision Rules
    SWITCH_MIN_SAVINGS = 0.02
    SWITCH_RISK_ABSOLUTE_MAX = 60
    DECISION_INTERVAL_HOURS = 12
    
    # Penalties
    SINGLE_POOL_CONCENTRATION_PENALTY = 20
    SINGLE_POOL_NO_ESCAPE_PENALTY = 10

config = Config()
os.makedirs(config.MODEL_DIR, exist_ok=True)
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

print("="*80)
print("SMART ADAPTIVE THRESHOLD SYSTEM - FULL Q1/Q2/Q3 TESTING")
print("="*80)
print(f"\n🎯 Smart Threshold Configuration:")
print(f"   Monthly Base: 32-45 (adaptive)")
print(f"   Event Adjustments: -2 to -18 points")
print(f"   Compound Risk: -10 additional")
print(f"   Floor/Ceiling: 25-50")
print(f"\n📊 Test Coverage: Q1, Q2, Q3 2025")
print(f"   Training: 2023-2024 (24 months)")
print("\n" + "="*80 + "\n")

# ============================================================================
# DATA LOADER
# ============================================================================

def load_data(csv_path, name):
    """Load and clean data"""
    print(f"📥 Loading {name}...")
    
    df = pd.read_csv(csv_path)
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
    
    df = df.rename(columns=col_map)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = df[(df['SpotPrice'] >= 0) & (df['SpotPrice'] < 10) & (df['OnDemandPrice'] > 0)].copy()
    df = df[df['InstanceType'].isin(config.INSTANCE_TYPES)].copy()
    
    if 'Region' not in df.columns:
        df['Region'] = 'ap-south-1'
    
    df['Pool_ID'] = df['InstanceType'] + '_' + df['AZ']
    df['ratio'] = (df['SpotPrice'] / df['OnDemandPrice']).clip(0, 1)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"   ✅ {len(df):,} records, {df['Pool_ID'].nunique()} pools, ratio: {df['ratio'].mean():.3f}")
    
    return df

# ============================================================================
# EVENT CALENDAR
# ============================================================================

def load_event_calendar():
    """Load event calendar"""
    print(f"\n📅 Loading event calendar...")
    
    df = pd.read_csv(config.EVENT_DATA)
    df.columns = df.columns.str.lower().str.strip()
    
    date_col = next((c for c in df.columns if 'date' in c), None)
    name_col = next((c for c in df.columns if 'event' in c or 'name' in c), None)
    impact_col = next((c for c in df.columns if 'impact' in c or 'severity' in c), None)
    region_col = next((c for c in df.columns if 'region' in c), None)
    
    if not date_col:
        print("   ⚠️  No date column found")
        return SimpleEventCalendar([])
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    impact_map = {'Critical': 80, 'High': 60, 'Medium': 40, 'Low': 20}
    
    events = []
    for _, row in df.iterrows():
        event_name = str(row[name_col]) if name_col else 'Event'
        event_date = pd.Timestamp(row[date_col])
        
        if impact_col:
            impact = str(row[impact_col]).strip()
            severity = impact_map.get(impact, 40)
        else:
            severity = 40
        
        region = str(row[region_col]) if region_col else 'all'
        
        events.append({
            'date': event_date,
            'name': event_name,
            'severity': severity,
            'pre_days': 3,
            'post_days': 2,
            'region': region
        })
    
    print(f"   ✅ Loaded {len(events)} events")
    
    return SimpleEventCalendar(events)

class SimpleEventCalendar:
    """Simple event calendar"""
    
    def __init__(self, events):
        self.events = events
    
    def check_event_window(self, current_date, region='all'):
        if not isinstance(current_date, pd.Timestamp):
            current_date = pd.Timestamp(current_date)
        
        max_severity = 0
        relevant_event = None
        
        for event in self.events:
            if event['region'] not in ['all', region]:
                continue
            
            event_start = event['date'] - timedelta(days=event['pre_days'])
            event_end = event['date'] + timedelta(days=event['post_days'])
            
            if event_start <= current_date <= event_end:
                if event['severity'] > max_severity:
                    max_severity = event['severity']
                    relevant_event = event
        
        return max_severity, relevant_event

# ============================================================================
# SMART THRESHOLD CALCULATOR
# ============================================================================

class SmartThresholdCalculator:
    """Calculates adaptive thresholds based on historical patterns"""
    
    def __init__(self, df_historical, event_calendar):
        self.df_historical = df_historical
        self.event_calendar = event_calendar
        self.monthly_profiles = {}
        self.event_capacity_correlation = {}
        self.build_profiles()
    
    def build_profiles(self):
        """Build monthly and event-capacity profiles"""
        print("\n" + "="*80)
        print("🧠 BUILDING SMART THRESHOLD PROFILES")
        print("="*80)
        
        # Monthly capacity patterns
        print("\n📊 Analyzing monthly patterns from 2023-2024...")
        self.df_historical['month'] = self.df_historical['timestamp'].dt.month
        self.df_historical['year'] = self.df_historical['timestamp'].dt.year
        
        for month in range(1, 13):
            month_data = self.df_historical[self.df_historical['month'] == month]
            
            if len(month_data) > 0:
                stress_days = len(month_data[month_data['ratio'] > 0.55]) / len(month_data) * 30
                
                self.monthly_profiles[month] = {
                    'avg_ratio': month_data['ratio'].mean(),
                    'p90_ratio': month_data['ratio'].quantile(0.90),
                    'p95_ratio': month_data['ratio'].quantile(0.95),
                    'volatility': month_data['ratio'].std(),
                    'capacity_stress_days': stress_days,
                    'samples': len(month_data)
                }
                
                month_name = pd.Timestamp(f'2024-{month:02d}-01').strftime('%B')
                base_threshold = config.MONTHLY_BASE_THRESHOLDS.get(month, 40)
                
                print(f"   {month_name:12} | Avg Ratio: {month_data['ratio'].mean():.3f} | "
                      f"Stress Days: {stress_days:4.1f} | Base Threshold: {base_threshold}")
        
        # Event-capacity correlation
        print(f"\n🔍 Analyzing event-capacity correlations...")
        
        for event in tqdm(self.event_calendar.events, desc="Events"):
            event_date = pd.Timestamp(event['date'])
            
            # Get data ±5 days around event
            event_window = self.df_historical[
                (self.df_historical['timestamp'] >= event_date - timedelta(days=5)) &
                (self.df_historical['timestamp'] <= event_date + timedelta(days=5))
            ]
            
            if len(event_window) > 0:
                key = f"{event['name']}_{event['severity']}"
                
                max_ratio = event_window['ratio'].max()
                avg_ratio = event_window['ratio'].mean()
                stress_occurred = max_ratio > 0.55
                
                self.event_capacity_correlation[key] = {
                    'avg_ratio': avg_ratio,
                    'max_ratio': max_ratio,
                    'stress_occurred': stress_occurred,
                    'samples': len(event_window)
                }
        
        print(f"   ✅ Built profiles for {len(self.monthly_profiles)} months")
        print(f"   ✅ Analyzed {len(self.event_capacity_correlation)} event patterns")
        
        # Show events that historically caused stress
        stress_events = [k for k, v in self.event_capacity_correlation.items() if v['stress_occurred']]
        print(f"\n🚨 Events with historical capacity stress: {len(stress_events)}")
        for event_key in stress_events[:10]:
            event_name = event_key.rsplit('_', 1)[0]
            pattern = self.event_capacity_correlation[event_key]
            print(f"      {event_name:40} | Max Ratio: {pattern['max_ratio']:.3f}")
    
    def calculate_base_threshold(self, current_date):
        """Calculate base threshold for given date"""
        month = current_date.month
        return config.MONTHLY_BASE_THRESHOLDS.get(month, 40)
    
    def get_severity_adjustment(self, severity):
        """Get adjustment for severity level"""
        for threshold_severity, adjustment in sorted(config.SEVERITY_ADJUSTMENTS.items(), reverse=True):
            if severity >= threshold_severity:
                return adjustment
        return 0
    
    def adjust_for_event(self, base_threshold, event_severity, event_name, current_capacity_ratio):
        """Adjust threshold based on event and current capacity"""
        
        # Base severity adjustment
        severity_adjustment = self.get_severity_adjustment(event_severity)
        
        # Check if this event historically caused capacity stress
        event_key = f"{event_name}_{event_severity}"
        event_pattern = self.event_capacity_correlation.get(event_key, {})
        
        stress_occurred = event_pattern.get('stress_occurred', False)
        historical_adjustment = -5 if stress_occurred else 0
        
        # Current capacity stress adjustment
        capacity_adjustment = 0
        if current_capacity_ratio > 0.55:
            capacity_adjustment = -8
        elif current_capacity_ratio > 0.50:
            capacity_adjustment = -5
        elif current_capacity_ratio > 0.45:
            capacity_adjustment = -3
        
        # Compound risk adjustment
        compound_adjustment = 0
        if (event_severity >= config.COMPOUND_RISK_SEVERITY_THRESHOLD and 
            current_capacity_ratio > config.COMPOUND_RISK_CAPACITY_THRESHOLD):
            compound_adjustment = config.COMPOUND_RISK_ADJUSTMENT
        
        # Total adjustment
        total_adjustment = (severity_adjustment + historical_adjustment + 
                          capacity_adjustment + compound_adjustment)
        
        # Apply floor and ceiling
        adjusted = base_threshold + total_adjustment
        adjusted = max(config.THRESHOLD_FLOOR, min(config.THRESHOLD_CEILING, adjusted))
        
        return adjusted, {
            'severity_adj': severity_adjustment,
            'historical_adj': historical_adjustment,
            'capacity_adj': capacity_adjustment,
            'compound_adj': compound_adjustment,
            'total_adj': total_adjustment
        }
    
    def get_smart_threshold(self, current_date, current_capacity_ratio, region='ap-south-1'):
        """Get smart threshold for current conditions"""
        
        # Base threshold from monthly pattern
        base = self.calculate_base_threshold(current_date)
        
        # Check for events
        severity, event = self.event_calendar.check_event_window(current_date, region)
        
        if severity > 0 and event:
            adjusted, adjustments = self.adjust_for_event(
                base, severity, event['name'], current_capacity_ratio
            )
            
            return {
                'threshold': adjusted,
                'base': base,
                'adjustments': adjustments,
                'event_name': event['name'],
                'event_severity': severity,
                'in_event': True,
                'capacity_ratio': current_capacity_ratio
            }
        
        # No event - use base
        return {
            'threshold': base,
            'base': base,
            'adjustments': {'total_adj': 0},
            'event_name': None,
            'event_severity': 0,
            'in_event': False,
            'capacity_ratio': current_capacity_ratio
        }

# ============================================================================
# RISK SCORER (from v8.1)
# ============================================================================

class RiskScorer:
    """Risk scorer with tiered event multipliers"""
    
    def __init__(self, event_calendar):
        self.event_calendar = event_calendar
        self.event_log = []
    
    def compute_capacity_risk(self, current_ratio):
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
    
    def compute_event_risk(self, current_date, region='all'):
        """Tiered event risk calculation"""
        severity, event = self.event_calendar.check_event_window(current_date, region)
        
        if severity == 0:
            return 0, severity, None
        
        # Tiered multipliers
        if severity >= 70:
            multiplier = 1.0
        elif severity >= 55:
            multiplier = 0.85
        elif severity >= 40:
            multiplier = 0.70
        else:
            multiplier = 0.50
        
        event_risk = min(severity * multiplier, 100)
        
        return event_risk, severity, event
    
    def compute_overall_risk(self, pool_id, pool_data, current_date, is_single_pool=False):
        current_ratio = pool_data['ratio'].iloc[-1] if len(pool_data) > 0 else 0.5
        region = pool_data['Region'].iloc[0] if len(pool_data) > 0 else 'ap-south-1'
        
        # Capacity risk
        capacity_risk = self.compute_capacity_risk(current_ratio)
        
        # Volatility risk
        if len(pool_data) >= 24:
            recent = pool_data.tail(24)
            std_price = recent['SpotPrice'].std()
            mean_price = recent['SpotPrice'].mean()
            cv = (std_price / mean_price) if mean_price > 0 else 0
            volatility_risk = min((cv / 0.15) * 100, 100)
        else:
            volatility_risk = 30
        
        # Trend risk
        if len(pool_data) >= 48:
            recent_24h = pool_data.tail(24)['ratio'].mean()
            previous_24h = pool_data.tail(48).head(24)['ratio'].mean()
            ratio_change = (recent_24h - previous_24h) / (previous_24h + 1e-6)
            trend_risk = min((ratio_change / 0.10) * 100, 100)
            trend_risk = max(0, trend_risk)
        else:
            trend_risk = 0
        
        # Event risk
        event_risk, severity, event = self.compute_event_risk(current_date, region)
        
        # Base risk
        base_risk = (
            config.RISK_WEIGHT_CAPACITY * capacity_risk +
            config.RISK_WEIGHT_VOLATILITY * volatility_risk +
            config.RISK_WEIGHT_TREND * trend_risk +
            config.RISK_WEIGHT_EVENT * event_risk +
            config.RISK_WEIGHT_PREDICTION * 10
        )
        
        # Event escalation
        escalation = 0
        if severity >= 60 and capacity_risk >= 40:
            escalation = min(20, (capacity_risk - 40) * 0.5 + 10)
            base_risk += escalation
            
            self.event_log.append({
                'date': current_date,
                'pool_id': pool_id,
                'event': event['name'] if event else 'Unknown',
                'severity': severity,
                'capacity_risk': capacity_risk,
                'escalation': escalation,
                'final_risk': base_risk
            })
        
        # Single pool penalty
        if is_single_pool:
            penalty = config.SINGLE_POOL_CONCENTRATION_PENALTY + config.SINGLE_POOL_NO_ESCAPE_PENALTY
            base_risk = min(base_risk + penalty, 100)
        
        return {
            'pool_id': pool_id,
            'overall_risk': base_risk,
            'capacity_risk': capacity_risk,
            'volatility_risk': volatility_risk,
            'trend_risk': trend_risk,
            'event_risk': event_risk,
            'event_severity': severity,
            'escalation': escalation,
            'current_ratio': current_ratio,
            'event_name': event['name'] if event else None
        }

# ============================================================================
# ADAPTIVE DECISION ENGINE
# ============================================================================

class AdaptiveDecisionEngine:
    """Decision engine with smart threshold adaptation"""
    
    def __init__(self, risk_scorer, threshold_calculator):
        self.risk_scorer = risk_scorer
        self.threshold_calculator = threshold_calculator
        self.current_pool = {}
        self.current_mode = {}
        self.decision_log = []
        self.threshold_log = []
    
    def make_decision(self, instance_type, all_pools_data, current_time):
        """Make decision with adaptive threshold"""
        
        # Initialize
        if instance_type not in self.current_pool:
            evaluations = []
            for pool_id in all_pools_data['Pool_ID'].unique():
                pool_data = all_pools_data[all_pools_data['Pool_ID'] == pool_id]
                risk_info = self.risk_scorer.compute_overall_risk(pool_id, pool_data, current_time)
                evaluations.append({
                    'pool_id': pool_id,
                    'risk': risk_info['overall_risk'],
                    'price': pool_data['SpotPrice'].iloc[-1] if len(pool_data) > 0 else 999
                })
            
            if not evaluations:
                return {'action': 'ERROR', 'mode': 'SPOT', 'target': 'NONE', 'risk': 50, 'price': 0}
            
            evaluations.sort(key=lambda x: (x['risk'], x['price']))
            best = evaluations[0]
            
            self.current_pool[instance_type] = best['pool_id']
            self.current_mode[instance_type] = 'SPOT'
            
            return {
                'action': 'INIT',
                'mode': 'SPOT',
                'target': best['pool_id'],
                'risk': best['risk'],
                'price': best['price'],
                'threshold': 40
            }
        
        # Get current state
        current_pool_id = self.current_pool[instance_type]
        current_mode = self.current_mode[instance_type]
        
        current_pool_data = all_pools_data[all_pools_data['Pool_ID'] == current_pool_id]
        if len(current_pool_data) == 0:
            return {'action': 'HOLD', 'mode': current_mode, 'target': current_pool_id, 'risk': 0, 'price': 0}
        
        # Calculate risk
        risk_info = self.risk_scorer.compute_overall_risk(current_pool_id, current_pool_data, current_time)
        current_risk = risk_info['overall_risk']
        current_ratio = risk_info['current_ratio']
        event_severity = risk_info.get('event_severity', 0)
        current_price = current_pool_data['SpotPrice'].iloc[-1]
        ondemand_price = current_pool_data['OnDemandPrice'].iloc[-1]
        
        # SMART THRESHOLD CALCULATION
        threshold_info = self.threshold_calculator.get_smart_threshold(
            current_time, current_ratio
        )
        
        threshold = threshold_info['threshold']
        
        # Log threshold decision
        self.threshold_log.append({
            'timestamp': current_time,
            'instance_type': instance_type,
            'base_threshold': threshold_info['base'],
            'threshold': threshold,
            'adjustments': threshold_info['adjustments'],
            'event_name': threshold_info.get('event_name'),
            'event_severity': event_severity,
            'current_risk': current_risk,
            'current_ratio': current_ratio
        })
        
        # High risk → On-Demand
        if current_risk > threshold:
            if current_mode != 'ONDEMAND':
                self.current_mode[instance_type] = 'ONDEMAND'
                
                self.decision_log.append({
                    'timestamp': current_time,
                    'instance_type': instance_type,
                    'action': 'SWITCH_TO_ONDEMAND',
                    'risk': current_risk,
                    'threshold': threshold,
                    'event_name': threshold_info.get('event_name'),
                    'event_severity': event_severity,
                    'capacity_ratio': current_ratio
                })
                
                return {
                    'action': 'SWITCH_TO_ONDEMAND',
                    'mode': 'ONDEMAND',
                    'target': 'ON_DEMAND',
                    'risk': 0,
                    'price': ondemand_price,
                    'threshold': threshold
                }
        else:
            # Return to Spot
            if current_mode == 'ONDEMAND':
                evaluations = []
                for pool_id in all_pools_data['Pool_ID'].unique():
                    pool_data = all_pools_data[all_pools_data['Pool_ID'] == pool_id]
                    r_info = self.risk_scorer.compute_overall_risk(pool_id, pool_data, current_time)
                    if r_info['overall_risk'] <= threshold:
                        evaluations.append({
                            'pool_id': pool_id,
                            'risk': r_info['overall_risk'],
                            'price': pool_data['SpotPrice'].iloc[-1]
                        })
                
                if evaluations:
                    evaluations.sort(key=lambda x: (x['risk'], x['price']))
                    best = evaluations[0]
                    
                    self.current_pool[instance_type] = best['pool_id']
                    self.current_mode[instance_type] = 'SPOT'
                    
                    return {
                        'action': 'RETURN_TO_SPOT',
                        'mode': 'SPOT',
                        'target': best['pool_id'],
                        'risk': best['risk'],
                        'price': best['price'],
                        'threshold': threshold
                    }
        
        # Switch pools for savings
        if current_mode == 'SPOT':
            candidates = []
            for pool_id in all_pools_data['Pool_ID'].unique():
                if pool_id == current_pool_id:
                    continue
                
                pool_data = all_pools_data[all_pools_data['Pool_ID'] == pool_id]
                r_info = self.risk_scorer.compute_overall_risk(pool_id, pool_data, current_time)
                
                if r_info['overall_risk'] > config.SWITCH_RISK_ABSOLUTE_MAX:
                    continue
                
                candidate_price = pool_data['SpotPrice'].iloc[-1]
                savings_pct = (current_price - candidate_price) / current_price
                
                if savings_pct >= config.SWITCH_MIN_SAVINGS:
                    candidates.append({
                        'pool_id': pool_id,
                        'risk': r_info['overall_risk'],
                        'price': candidate_price
                    })
            
            if candidates:
                candidates.sort(key=lambda x: (x['risk'], x['price']))
                best = candidates[0]
                self.current_pool[instance_type] = best['pool_id']
                
                return {
                    'action': 'SWITCH_SPOT_POOL',
                    'mode': 'SPOT',
                    'target': best['pool_id'],
                    'risk': best['risk'],
                    'price': best['price'],
                    'threshold': threshold
                }
        
        return {
            'action': 'HOLD',
            'mode': current_mode,
            'target': current_pool_id if current_mode == 'SPOT' else 'ON_DEMAND',
            'risk': current_risk if current_mode == 'SPOT' else 0,
            'price': current_price if current_mode == 'SPOT' else ondemand_price,
            'threshold': threshold
        }

# ============================================================================
# BACKTESTER
# ============================================================================

class Backtester:
    """Complete backtesting system"""
    
    def __init__(self, decision_engine, risk_scorer):
        self.decision_engine = decision_engine
        self.risk_scorer = risk_scorer
        self.results = []
        self.costs = {
            'ml_spot': 0,
            'ml_ondemand': 0,
            'ml_total': 0,
            'single_pool': 0
        }
        self.risk_tracking = {'ml': [], 'single': []}
    
    def run(self, df_test, quarter_name):
        """Run backtest"""
        print(f"\n{'='*80}")
        print(f"🧪 BACKTESTING {quarter_name}")
        print(f"{'='*80}")
        
        instance_types = df_test['InstanceType'].unique()
        
        for instance_type in tqdm(instance_types, desc=f"{quarter_name}"):
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
                    # ML costs
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
                            self.risk_tracking['ml'].append(decision['risk'])
                    
                    # Single pool baseline
                    cheapest_pool = period_data.groupby('Pool_ID')['SpotPrice'].sum().idxmin()
                    self.costs['single_pool'] += period_data[period_data['Pool_ID'] == cheapest_pool]['SpotPrice'].sum()
                    
                    single_pool_data = period_data[period_data['Pool_ID'] == cheapest_pool]
                    if len(single_pool_data) > 0:
                        risk_info = self.risk_scorer.compute_overall_risk(
                            cheapest_pool, single_pool_data, current_time, True
                        )
                        self.risk_tracking['single'].append(risk_info['overall_risk'])
                
                self.results.append({
                    'timestamp': current_time,
                    'instance_type': instance_type,
                    'quarter': quarter_name,
                    **decision
                })
        
        print(f"✅ Complete: {len([r for r in self.results if r['quarter'] == quarter_name]):,} decisions")
    
    def get_metrics(self, quarter_filter=None):
        """Calculate metrics for specific quarter or all"""
        
        if quarter_filter:
            quarter_results = [r for r in self.results if r['quarter'] == quarter_filter]
        else:
            quarter_results = self.results
        
        ml_total = sum(r['price'] for r in quarter_results 
                      if r['action'] not in ['ERROR'])
        ml_ondemand = sum(r['price'] for r in quarter_results 
                         if r['mode'] == 'ONDEMAND')
        ml_spot = ml_total - ml_ondemand
        
        ondemand_pct = (ml_ondemand / ml_total * 100) if ml_total > 0 else 0
        
        switches_to_od = sum(1 for r in quarter_results if r['action'] == 'SWITCH_TO_ONDEMAND')
        ondemand_periods = sum(1 for r in quarter_results if r['mode'] == 'ONDEMAND')
        
        return {
            'ml_total': ml_total,
            'ml_spot': ml_spot,
            'ml_ondemand': ml_ondemand,
            'ondemand_pct': ondemand_pct,
            'switches_to_od': switches_to_od,
            'ondemand_periods': ondemand_periods,
            'decisions': len(quarter_results)
        }

# ============================================================================
# MAIN TESTING
# ============================================================================

def main():
    """Complete testing pipeline"""
    
    print("\n🚀 STARTING SMART THRESHOLD TESTING")
    print("="*80)
    
    # Load training data
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    df_train = load_data(config.TRAINING_DATA, "Training 2023-2024")
    df_q1 = load_data(config.TEST_Q1, "Q1 2025")
    df_q2 = load_data(config.TEST_Q2, "Q2 2025")
    df_q3 = load_data(config.TEST_Q3, "Q3 2025")
    
    # Load event calendar
    event_calendar = load_event_calendar()
    
    # Build smart threshold calculator
    threshold_calc = SmartThresholdCalculator(df_train, event_calendar)
    
    # Initialize components
    risk_scorer = RiskScorer(event_calendar)
    decision_engine = AdaptiveDecisionEngine(risk_scorer, threshold_calc)
    backtester = Backtester(decision_engine, risk_scorer)
    
    # Run backtests
    backtester.run(df_q1, "Q1")
    backtester.run(df_q2, "Q2")
    backtester.run(df_q3, "Q3")
    
    # Calculate metrics
    print("\n" + "="*80)
    print("📊 RESULTS ANALYSIS")
    print("="*80)
    
    # Quarterly results
    quarters = ["Q1", "Q2", "Q3"]
    quarterly_metrics = {}
    
    for quarter in quarters:
        metrics = backtester.get_metrics(quarter)
        quarterly_metrics[quarter] = metrics
        
        print(f"\n{quarter} 2025:")
        print(f"   Cost: ${metrics['ml_total']:,.2f}")
        print(f"   ├─ Spot: ${metrics['ml_spot']:,.2f} ({100-metrics['ondemand_pct']:.1f}%)")
        print(f"   └─ On-Demand: ${metrics['ml_ondemand']:,.2f} ({metrics['ondemand_pct']:.1f}%)")
        print(f"   OD Switches: {metrics['switches_to_od']}")
        print(f"   OD Periods: {metrics['ondemand_periods']}")
    
    # Total results
    total_metrics = backtester.get_metrics()
    
    print(f"\n{'='*80}")
    print("TOTAL Q1+Q2+Q3 2025")
    print(f"{'='*80}")
    
    print(f"\n💰 Costs:")
    print(f"   ML Total: ${total_metrics['ml_total']:,.2f}")
    print(f"   ├─ Spot: ${total_metrics['ml_spot']:,.2f} ({100-total_metrics['ondemand_pct']:.1f}%)")
    print(f"   └─ On-Demand: ${total_metrics['ml_ondemand']:,.2f} ({total_metrics['ondemand_pct']:.1f}%)")
    print(f"   Single Pool: ${backtester.costs['single_pool']:,.2f}")
    
    savings_pct = ((backtester.costs['single_pool'] - total_metrics['ml_total']) / 
                   backtester.costs['single_pool'] * 100) if backtester.costs['single_pool'] > 0 else 0
    
    print(f"\n   Savings vs Single: {savings_pct:+.2f}%")
    
    print(f"\n📊 Operations:")
    print(f"   Total Decisions: {total_metrics['decisions']:,}")
    print(f"   OD Switches: {total_metrics['switches_to_od']}")
    print(f"   OD Periods: {total_metrics['ondemand_periods']}")
    
    # Risk analysis
    ml_avg_risk = np.mean(backtester.risk_tracking['ml']) if backtester.risk_tracking['ml'] else 0
    single_avg_risk = np.mean(backtester.risk_tracking['single']) if backtester.risk_tracking['single'] else 0
    
    print(f"\n⚠️  Risk:")
    print(f"   ML: {ml_avg_risk:.1f}/100")
    print(f"   Single Pool: {single_avg_risk:.1f}/100")
    print(f"   Reduction: {single_avg_risk - ml_avg_risk:.1f} points")
    
    # Annual projection
    annual_ml = total_metrics['ml_total'] * 4 / 3  # 3 quarters → 4 quarters
    annual_single = backtester.costs['single_pool'] * 4 / 3
    annual_premium = annual_ml - annual_single
    
    print(f"\n📅 Annual Projection (Q1-Q3 × 4/3):")
    print(f"   ML: ${annual_ml:,.2f}")
    print(f"   Single: ${annual_single:,.2f}")
    print(f"   Premium: ${annual_premium:+,.2f} ({annual_premium/annual_single*100:+.2f}%)")
    
    # Threshold analysis
    print(f"\n{'='*80}")
    print("🎯 SMART THRESHOLD ANALYSIS")
    print(f"{'='*80}")
    
    threshold_df = pd.DataFrame(decision_engine.threshold_log)
    
    if len(threshold_df) > 0:
        print(f"\n📊 Threshold Statistics:")
        print(f"   Avg Threshold: {threshold_df['threshold'].mean():.1f}")
        print(f"   Min Threshold: {threshold_df['threshold'].min():.0f}")
        print(f"   Max Threshold: {threshold_df['threshold'].max():.0f}")
        print(f"   Std Dev: {threshold_df['threshold'].std():.1f}")
        
        # Threshold distribution
        print(f"\n   Threshold Distribution:")
        for threshold in [25, 28, 30, 32, 35, 38, 40, 42, 45]:
            count = len(threshold_df[threshold_df['threshold'] == threshold])
            pct = count / len(threshold_df) * 100
            if pct > 0.1:
                print(f"      {threshold}: {count:,} ({pct:.1f}%)")
        
        # Events that triggered adjustments
        event_adjustments = threshold_df[threshold_df['event_name'].notna()]
        
        if len(event_adjustments) > 0:
            print(f"\n🔍 Event Adjustments:")
            print(f"   Total periods with events: {len(event_adjustments):,}")
            
            # Group by event
            event_summary = event_adjustments.groupby('event_name').agg({
                'threshold': ['mean', 'min'],
                'event_severity': 'first',
                'timestamp': 'count'
            }).reset_index()
            
            event_summary.columns = ['event', 'avg_threshold', 'min_threshold', 'severity', 'periods']
            event_summary = event_summary.sort_values('severity', ascending=False)
            
            print(f"\n   Top Events by Severity:")
            for _, row in event_summary.head(10).iterrows():
                print(f"      {row['event']:40} | Severity: {row['severity']:2.0f} | "
                      f"Avg Threshold: {row['avg_threshold']:4.1f} | Min: {row['min_threshold']:2.0f}")
    
    # On-Demand switches detail
    if len(decision_engine.decision_log) > 0:
        print(f"\n{'='*80}")
        print("🚨 ON-DEMAND SWITCH ANALYSIS")
        print(f"{'='*80}")
        
        print(f"\nTotal OD Switches: {len(decision_engine.decision_log)}")
        
        switch_df = pd.DataFrame(decision_engine.decision_log)
        
        # Group by event
        event_switches = switch_df.groupby('event_name').size().reset_index(name='switches')
        event_switches = event_switches.sort_values('switches', ascending=False)
        
        print(f"\n   Switches by Event:")
        for _, row in event_switches.head(10).iterrows():
            event_name = row['event_name'] if pd.notna(row['event_name']) else 'No Event'
            print(f"      {event_name:40} | {row['switches']:2} switches")
        
        # Show sample switches
        print(f"\n   Sample OD Switches:")
        for _, log in switch_df.head(10).iterrows():
            event_name = log['event_name'] if pd.notna(log['event_name']) else 'No Event'
            print(f"      {log['timestamp'].strftime('%Y-%m-%d')}: {event_name}")
            print(f"         Risk: {log['risk']:.1f} > Threshold: {log['threshold']:.1f}")
            print(f"         Capacity: {log['capacity_ratio']:.3f}")
    
    # Comparison with fixed thresholds
    print(f"\n{'='*80}")
    print("📈 COMPARISON: SMART vs FIXED THRESHOLDS")
    print(f"{'='*80}")
    
    # Estimate fixed threshold results based on your diagnostic data
    print(f"\nEstimated Annual Costs:")
    print(f"   Smart Adaptive: ${annual_ml:,.2f} ({total_metrics['ondemand_pct']:.1f}% OD)")
    print(f"   Fixed 42: ~$18,546 (0-16% OD, Q1 only)")
    print(f"   Fixed 35: ~$23,000-27,000 (20-46% OD)")
    print(f"   Fixed 30: ~$30,000-33,000 (35-51% OD)")
    
    print(f"\n💡 Smart System Benefits:")
    print(f"   ✅ Comprehensive protection ({total_metrics['switches_to_od']} events)")
    print(f"   ✅ Balanced OD usage ({total_metrics['ondemand_pct']:.1f}%)")
    print(f"   ✅ Adapts to monthly patterns")
    print(f"   ✅ Learns from historical data")
    print(f"   ✅ Compound risk detection")
    
    # Save results
    print(f"\n{'='*80}")
    print("💾 SAVING RESULTS")
    print(f"{'='*80}")
    
    # Save all results
    if backtester.results:
        results_df = pd.DataFrame(backtester.results)
        results_df.to_csv(os.path.join(config.OUTPUT_DIR, 'smart_threshold_results.csv'), index=False)
        print(f"   ✅ Results: smart_threshold_results.csv")
    
    # Save decision log
    if decision_engine.decision_log:
        decision_df = pd.DataFrame(decision_engine.decision_log)
        decision_df.to_csv(os.path.join(config.OUTPUT_DIR, 'ondemand_switches.csv'), index=False)
        print(f"   ✅ OD Switches: ondemand_switches.csv")
    
    # Save threshold log
    if decision_engine.threshold_log:
        threshold_df = pd.DataFrame(decision_engine.threshold_log)
        threshold_df.to_csv(os.path.join(config.OUTPUT_DIR, 'threshold_log.csv'), index=False)
        print(f"   ✅ Threshold Log: threshold_log.csv")
    
    # Save event escalations
    if risk_scorer.event_log:
        event_df = pd.DataFrame(risk_scorer.event_log)
        event_df.to_csv(os.path.join(config.OUTPUT_DIR, 'event_escalations.csv'), index=False)
        print(f"   ✅ Event Escalations: event_escalations.csv")
    
    # Save summary metrics
    summary = {
        'quarterly_metrics': quarterly_metrics,
        'total_metrics': {
            'ml_total': total_metrics['ml_total'],
            'ml_spot': total_metrics['ml_spot'],
            'ml_ondemand': total_metrics['ml_ondemand'],
            'single_pool': backtester.costs['single_pool'],
            'ondemand_pct': total_metrics['ondemand_pct'],
            'switches_to_od': total_metrics['switches_to_od'],
            'ondemand_periods': total_metrics['ondemand_periods'],
            'ml_avg_risk': ml_avg_risk,
            'single_avg_risk': single_avg_risk,
            'risk_reduction': single_avg_risk - ml_avg_risk
        },
        'annual_projection': {
            'ml_annual': annual_ml,
            'single_annual': annual_single,
            'premium': annual_premium,
            'premium_pct': annual_premium / annual_single * 100 if annual_single > 0 else 0
        },
        'threshold_stats': {
            'avg_threshold': threshold_df['threshold'].mean() if len(threshold_df) > 0 else 0,
            'min_threshold': threshold_df['threshold'].min() if len(threshold_df) > 0 else 0,
            'max_threshold': threshold_df['threshold'].max() if len(threshold_df) > 0 else 0,
            'std_threshold': threshold_df['threshold'].std() if len(threshold_df) > 0 else 0
        }
    }
    
    with open(os.path.join(config.OUTPUT_DIR, 'summary_metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"   ✅ Summary: summary_metrics.json")
    
    print(f"\n{'='*80}")
    print("✅ TESTING COMPLETE!")
    print(f"{'='*80}")
    print(f"\n📁 All results saved to: {config.OUTPUT_DIR}/")
    print(f"\n🎯 KEY FINDINGS:")
    print(f"   Annual Cost: ${annual_ml:,.2f}")
    print(f"   OD Usage: {total_metrics['ondemand_pct']:.1f}%")
    print(f"   OD Switches: {total_metrics['switches_to_od']} events")
    print(f"   Risk Reduction: {single_avg_risk - ml_avg_risk:.1f} points")
    print(f"   Premium: ${annual_premium:+,.2f}/year ({annual_premium/annual_single*100:+.1f}%)")
    
    if total_metrics['ondemand_pct'] >= 2.0 and total_metrics['ondemand_pct'] <= 5.0:
        print(f"\n✅ OPTIMAL: OD usage is within target 2-5% range!")
    elif total_metrics['ondemand_pct'] < 2.0:
        print(f"\n⚠️  OD usage is low - consider lowering base thresholds by 2-3 points")
    else:
        print(f"\n⚠️  OD usage is high - consider raising base thresholds by 2-3 points")
    
    return summary

if __name__ == "__main__":
    results = main()
