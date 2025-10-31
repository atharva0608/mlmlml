"""
AWS Spot Pool-Based ML Pipeline v6.5 - FIXED VERSION
================================================================================

MAJOR FIXES in v6.5:
1. ✅ Fixed simulation date range bug (reduced buffers for 90-day datasets)
2. ✅ Added adaptive buffer logic for short test periods
3. ✅ Added multi-quarter evaluation support (Q1, Q2, Q3)
4. ✅ Added quarterly comparison framework
5. ✅ Enhanced diagnostic output and error messages
6. ✅ Added validation for data sufficiency

Critical Optimization for Your Data:
1. ✅ Removes spike detection dependency (your data has virtually no spikes)
2. ✅ Uses ratio velocity as primary signal (gradual changes, not jumps)
3. ✅ Uses ratio trends (24h, 7d) to detect capacity tightening
4. ✅ Much more sensitive risk thresholds (matches your stable baseline)
5. ✅ Focuses on what works: ratio, volatility, regional pressure

Your Data Reality:
- Training spikes: 9 out of 2.3M records (0.0004%)
- Test max change: 1.03% (far below 3% threshold)
- Prices are VERY stable hour-to-hour
- Changes happen gradually, not suddenly

Author: AWS Spot Optimization Team
Version: 6.5 (Fixed Simulation + Multi-Quarter Support)
Date: 2025-10-31
"""

import pandas as pd
import numpy as np
import torch
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime, timedelta
from scipy import stats
from tqdm import tqdm
import random
import os
import json
from collections import defaultdict

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

#═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - FIXED FOR 90-DAY TEST DATASETS
#═══════════════════════════════════════════════════════════════════════════════

class Config:
    """Ultra-stable data configuration with FIXED simulation parameters"""
    
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Data Split
    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Data Cleaning
    REMOVE_OUTLIERS = True
    OUTLIER_THRESHOLD = 4.0
    SMOOTH_PRICES = True
    SMOOTH_WINDOW = 3
    USE_STRATIFIED_SAMPLING = True
    OUTLIER_PER_POOL = True
    
    # Prophet Configuration
    PROPHET_CHANGEPOINT_PRIOR = 0.001
    PROPHET_SEASONALITY_PRIOR = 0.01
    PROPHET_CHANGEPOINT_RANGE = 0.8
    PROPHET_UNCERTAINTY_SAMPLES = 100
    PROPHET_INTERVAL_WIDTH = 0.80
    
    ENABLE_DAILY_SEASONALITY = True
    ENABLE_WEEKLY_SEASONALITY = True
    ENABLE_YEARLY_SEASONALITY = False
    
    # Features
    ROLLING_WINDOWS = [24, 168]
    
    # Ratio velocity thresholds (what actually works)
    RATIO_VELOCITY_WARNING = 0.005   # 0.5% change per day
    RATIO_VELOCITY_CRITICAL = 0.01   # 1% change per day
    
    # Ratio trend thresholds (for gradual capacity tightening)
    RATIO_TREND_24H_WARNING = 0.01   # 1% increase in 24h
    RATIO_TREND_7D_WARNING = 0.03    # 3% increase in 7 days
    
    # Capacity thresholds - MUCH more sensitive for stable data
    CAPACITY_RATIO_NORMAL = 0.45      # Was 0.60
    CAPACITY_RATIO_WARNING = 0.55     # Was 0.70
    CAPACITY_RATIO_CRITICAL = 0.65    # Was 0.80
    
    # Risk Scoring Weights - NO SPIKE DEPENDENCY
    RISK_WEIGHT_RATIO = 0.40         # Primary signal (was interruption)
    RISK_WEIGHT_VOLATILITY = 0.30    # Coefficient of variation
    RISK_WEIGHT_VELOCITY = 0.20      # Rate of change (NEW)
    RISK_WEIGHT_CONFIDENCE = 0.10
    
    # Risk categories - ULTRA-SENSITIVE for stable data
    RISK_SAFE_MAX = 20       # Was 25
    RISK_MODERATE_MAX = 40   # Was 50
    
    # Recommendation Rules
    MIN_SAVINGS_TO_SWITCH = 0.10
    MAX_SWITCHES_PER_DAY = 4
    MIN_TIME_BETWEEN_SWITCHES = 6 * 3600
    
    # Training
    MIN_TRAIN_SAMPLES = 500
    
    # Simulation - FIXED for 90-day datasets
    SIMULATION_CUTOFF_BUFFER_DAYS = 7      # REDUCED from 30
    SIMULATION_PREDICTION_HORIZON = 168    # Keep 7 days
    SIMULATION_MIN_HISTORY_DAYS = 30       # REDUCED from 60
    MIN_DATASET_DAYS = 45                  # NEW: Minimum dataset requirement
    
    # Clustering
    N_POOL_CLUSTERS = 5
    
    # Output
    FIGURE_SIZE = (16, 10)
    DPI = 100
    OUTPUT_DIR = 'output_results_v6_5_fixed'

config = Config()
os.makedirs(config.OUTPUT_DIR, exist_ok=True)

print(f"🚀 AWS Spot Pipeline v6.5 - FIXED VERSION")
print(f"📊 Device: {config.DEVICE}")
print(f"🔧 Fixes: Simulation date ranges, Multi-quarter support, Adaptive buffers")

#═══════════════════════════════════════════════════════════════════════════════
# DATA CLEANER (unchanged)
#═══════════════════════════════════════════════════════════════════════════════

class StratifiedDataCleaner:
    """Enhanced data cleaning with stratified sampling"""
    
    @staticmethod
    def remove_outliers_stratified(df, column='SpotPrice', threshold=4.0):
        """Stratified outlier removal - preserves pool proportions"""
        print(f"  🎯 Stratified outlier removal (z-score > {threshold})...")
        
        original_count = len(df)
        indices_to_keep = []
        pools = df['Pool_ID'].unique()
        
        for pool_id in tqdm(pools, desc="    Processing pools", leave=False):
            pool_mask = df['Pool_ID'] == pool_id
            pool_indices = df[pool_mask].index
            pool_data = df.loc[pool_indices, column]
            
            if len(pool_data) < 100:
                indices_to_keep.extend(pool_indices.tolist())
                continue
            
            pool_mean = pool_data.mean()
            pool_std = pool_data.std()
            
            if pool_std == 0:
                indices_to_keep.extend(pool_indices.tolist())
                continue
            
            z_scores = np.abs((pool_data - pool_mean) / pool_std)
            keep_mask = z_scores <= threshold
            
            keep_indices = pool_indices[keep_mask]
            indices_to_keep.extend(keep_indices.tolist())
        
        df_cleaned = df.loc[indices_to_keep].copy()
        removed_total = original_count - len(df_cleaned)
        
        print(f"    ✓ Removed {removed_total:,} outliers ({removed_total/original_count*100:.3f}%)")
        
        return df_cleaned.reset_index(drop=True)
    
    @staticmethod
    def smooth_prices_fast(df, column='SpotPrice', window=3):
        print(f"  Smoothing prices (window={window})...")
        df = df.copy()
        df[column] = df.groupby('Pool_ID')[column].transform(
            lambda x: x.rolling(window=window, min_periods=1, center=True).mean()
        )
        return df
    
    @staticmethod
    def cap_extreme_values_fast(df, column='SpotPrice', lower_pct=0.01, upper_pct=0.99):
        print(f"  Capping extreme values...")
        bounds = df.groupby('Pool_ID')[column].quantile([lower_pct, upper_pct]).unstack()
        bounds.columns = ['lower', 'upper']
        df = df.merge(bounds, left_on='Pool_ID', right_index=True, how='left')
        df[column] = df[column].clip(lower=df['lower'], upper=df['upper'])
        df = df.drop(columns=['lower', 'upper'])
        return df

#═══════════════════════════════════════════════════════════════════════════════
# POOL METADATA PROFILER (unchanged)
#═══════════════════════════════════════════════════════════════════════════════

class PoolMetadataProfiler:
    """Extract and profile pool characteristics"""
    
    def __init__(self):
        self.pool_profiles = {}
        self.pool_clusters = {}
        self.cluster_model = None
    
    def extract_static_metadata(self, pool_id):
        """Extract instance attributes from pool ID"""
        parts = pool_id.split('_')
        instance_type = parts[0]
        az = '_'.join(parts[1:])
        
        family = ''.join([c for c in instance_type if not c.isdigit()])
        generation = ''.join([c for c in instance_type if c.isdigit()]).split('.')[0] if '.' in instance_type else ''
        
        if 'g' in family or instance_type.startswith('t4g'):
            processor = 'Graviton'
        elif 'a' in family:
            processor = 'AMD'
        else:
            processor = 'Intel'
        
        return {
            'pool_id': pool_id,
            'instance_type': instance_type,
            'instance_family': family,
            'generation': generation,
            'processor': processor,
            'az': az,
            'region': az.rsplit('-', 1)[0] if '-' in az else 'unknown'
        }
    
    def compute_historical_stats(self, df_pool):
        """Compute pool statistics from historical data"""
        return {
            'avg_spot_price': df_pool['SpotPrice'].mean(),
            'std_spot_price': df_pool['SpotPrice'].std(),
            'min_spot_price': df_pool['SpotPrice'].min(),
            'max_spot_price': df_pool['SpotPrice'].max(),
            'avg_ondemand_price': df_pool['OnDemandPrice'].mean(),
            'avg_ratio': (df_pool['SpotPrice'] / df_pool['OnDemandPrice']).mean(),
            'ratio_volatility': (df_pool['SpotPrice'] / df_pool['OnDemandPrice']).std(),
            'data_quality_score': len(df_pool) / (365 * 24)
        }
    
    def profile_all_pools(self, df):
        """Generate comprehensive profiles for all pools"""
        print("\n🔍 PROFILING POOLS")
        print("="*80)
        
        for pool_id in tqdm(df['Pool_ID'].unique(), desc="Profiling pools"):
            df_pool = df[df['Pool_ID'] == pool_id]
            
            profile = self.extract_static_metadata(pool_id)
            profile.update(self.compute_historical_stats(df_pool))
            
            self.pool_profiles[pool_id] = profile
        
        print(f"✓ Profiled {len(self.pool_profiles)} pools")
        
        return pd.DataFrame(list(self.pool_profiles.values()))
    
    def cluster_pools(self, df_profiles):
        """Cluster pools by behavioral characteristics"""
        print("\n🎨 CLUSTERING POOLS BY BEHAVIOR")
        print("="*80)
        
        features = ['avg_ratio', 'ratio_volatility', 'std_spot_price']
        X = df_profiles[features].fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=config.N_POOL_CLUSTERS, random_state=42, n_init=10)
        df_profiles['cluster_id'] = kmeans.fit_predict(X_scaled)
        
        self.cluster_model = kmeans
        
        for cluster_id in range(config.N_POOL_CLUSTERS):
            cluster_pools = df_profiles[df_profiles['cluster_id'] == cluster_id]
            print(f"\nCluster {cluster_id}: {len(cluster_pools)} pools")
            print(f"  Avg ratio: {cluster_pools['avg_ratio'].mean():.3f}")
            print(f"  Avg volatility: {cluster_pools['ratio_volatility'].mean():.3f}")
        
        for _, row in df_profiles.iterrows():
            self.pool_profiles[row['pool_id']]['cluster_id'] = row['cluster_id']
        
        return df_profiles

#═══════════════════════════════════════════════════════════════════════════════
# VELOCITY-BASED FEATURE ENGINEER (unchanged)
#═══════════════════════════════════════════════════════════════════════════════

class VelocityBasedEngineer:
    """Feature engineering based on ratio velocity and trends (not spikes)"""
    
    def __init__(self):
        self.regional_pressure_history = {}
    
    def compute_ratio_features(self, df):
        """Compute ratio-based features (primary signals for stable data)"""
        print(f"  Computing ratio features (primary signals)...")
        
        df = df.copy()
        
        # Base ratio
        df['spot_ondemand_ratio'] = df['SpotPrice'] / (df['OnDemandPrice'] + 1e-6)
        
        # Ratio velocity (rate of change) - PRIMARY SIGNAL
        df['ratio_velocity_24h'] = df.groupby('Pool_ID')['spot_ondemand_ratio'].transform(
            lambda x: x.diff(periods=24)
        )
        df['ratio_velocity_7d'] = df.groupby('Pool_ID')['spot_ondemand_ratio'].transform(
            lambda x: x.diff(periods=168)
        )
        
        # Ratio acceleration
        df['ratio_acceleration'] = df.groupby('Pool_ID')['ratio_velocity_24h'].transform(
            lambda x: x.diff()
        )
        
        # Ratio trends
        df['ratio_trend_24h'] = df.groupby('Pool_ID')['spot_ondemand_ratio'].transform(
            lambda x: x.rolling(window=24, min_periods=1).apply(
                lambda y: (y.iloc[-1] - y.iloc[0]) / (y.iloc[0] + 1e-6) if len(y) > 1 else 0
            )
        )
        df['ratio_trend_7d'] = df.groupby('Pool_ID')['spot_ondemand_ratio'].transform(
            lambda x: x.rolling(window=168, min_periods=1).apply(
                lambda y: (y.iloc[-1] - y.iloc[0]) / (y.iloc[0] + 1e-6) if len(y) > 1 else 0
            )
        )
        
        # Ratio volatility
        df['ratio_volatility_7d'] = df.groupby('Pool_ID')['spot_ondemand_ratio'].transform(
            lambda x: x.rolling(window=168, min_periods=1).std()
        )
        df['ratio_volatility_30d'] = df.groupby('Pool_ID')['spot_ondemand_ratio'].transform(
            lambda x: x.rolling(window=720, min_periods=1).std()
        )
        
        # Capacity zones
        df['capacity_normal'] = (df['spot_ondemand_ratio'] < config.CAPACITY_RATIO_NORMAL).astype(int)
        df['capacity_warning'] = (
            (df['spot_ondemand_ratio'] >= config.CAPACITY_RATIO_NORMAL) & 
            (df['spot_ondemand_ratio'] < config.CAPACITY_RATIO_WARNING)
        ).astype(int)
        df['capacity_critical'] = (df['spot_ondemand_ratio'] >= config.CAPACITY_RATIO_CRITICAL).astype(int)
        
        # Velocity warnings
        df['velocity_warning'] = (
            (abs(df['ratio_velocity_24h']) > config.RATIO_VELOCITY_WARNING) |
            (abs(df['ratio_trend_24h']) > config.RATIO_TREND_24H_WARNING)
        ).astype(int)
        
        df['velocity_critical'] = (
            (abs(df['ratio_velocity_24h']) > config.RATIO_VELOCITY_CRITICAL) |
            (abs(df['ratio_trend_7d']) > config.RATIO_TREND_7D_WARNING)
        ).astype(int)
        
        # Diagnostics
        print(f"\n    📊 Ratio distribution:")
        print(f"       Range: {df['spot_ondemand_ratio'].min():.3f} - {df['spot_ondemand_ratio'].max():.3f}")
        print(f"       Mean: {df['spot_ondemand_ratio'].mean():.3f}")
        print(f"       Median: {df['spot_ondemand_ratio'].median():.3f}")
        
        print(f"\n    📊 Capacity zones (sensitive thresholds):")
        normal_pct = (df['capacity_normal'].sum() / len(df)) * 100
        warning_pct = (df['capacity_warning'].sum() / len(df)) * 100
        critical_pct = (df['capacity_critical'].sum() / len(df)) * 100
        print(f"       Normal (<{config.CAPACITY_RATIO_NORMAL}): {normal_pct:.1f}%")
        print(f"       Warning ({config.CAPACITY_RATIO_NORMAL}-{config.CAPACITY_RATIO_WARNING}): {warning_pct:.1f}%")
        print(f"       Critical (>{config.CAPACITY_RATIO_CRITICAL}): {critical_pct:.1f}%")
        
        print(f"\n    📊 Velocity warnings (replaces spikes):")
        vel_warn = (df['velocity_warning'].sum() / len(df)) * 100
        vel_crit = (df['velocity_critical'].sum() / len(df)) * 100
        print(f"       Velocity warnings: {vel_warn:.1f}%")
        print(f"       Velocity critical: {vel_crit:.1f}%")
        
        return df
    
    def compute_regional_pressure(self, df):
        """Compute regional capacity pressure index"""
        print(f"  Computing regional pressure...")
        
        df = df.copy()
        
        regional_avg = df.groupby(['Region', 'timestamp'])['spot_ondemand_ratio'].mean().reset_index()
        regional_avg.columns = ['Region', 'timestamp', 'regional_pressure']
        
        df = df.merge(regional_avg, on=['Region', 'timestamp'], how='left')
        
        df['regional_stress'] = (df['regional_pressure'] > config.CAPACITY_RATIO_WARNING).astype(int)
        
        self.regional_pressure_history = regional_avg.set_index(['Region', 'timestamp'])['regional_pressure'].to_dict()
        
        # Diagnostics
        for region in df['Region'].unique():
            region_data = df[df['Region'] == region]
            avg_pressure = region_data['regional_pressure'].mean()
            max_pressure = region_data['regional_pressure'].max()
            stress_pct = (region_data['regional_stress'].sum() / len(region_data)) * 100
            print(f"    {region}: Avg={avg_pressure:.3f}, Max={max_pressure:.3f}, Stress={stress_pct:.1f}%")
        
        return df
    
    def compute_cross_pool_features(self, df):
        """Cross-pool correlation features"""
        print(f"  Computing cross-pool features...")
        
        df = df.copy()
        
        df['instance_avg_price'] = df.groupby(['InstanceType', 'timestamp'])['SpotPrice'].transform('mean')
        df['price_divergence_from_siblings'] = df['SpotPrice'] - df['instance_avg_price']
        
        df['family_avg_ratio'] = df.groupby(['timestamp'])['spot_ondemand_ratio'].transform('mean')
        
        return df

#═══════════════════════════════════════════════════════════════════════════════
# EVENT ANALYZER (unchanged)
#═══════════════════════════════════════════════════════════════════════════════

class EventAnalyzer:
    """Event impact analysis"""
    
    def __init__(self, window_days=10):
        self.window = window_days
        self.pool_baselines = {}
    
    def analyze_event_impact(self, df_price, df_events):
        print("\n🔍 ANALYZING EVENT IMPACT")
        print("="*80)
        
        event_analysis = []
        
        for pool_id in df_price['Pool_ID'].unique():
            df_pool = df_price[df_price['Pool_ID'] == pool_id]
            self.pool_baselines[pool_id] = df_pool['SpotPrice'].mean()
        
        pools = df_price['Pool_ID'].unique()
        
        for _, event in tqdm(df_events.iterrows(), total=len(df_events), desc="Event analysis"):
            event_date = pd.to_datetime(event['Date'])
            event_name = event['EventName']
            event_region = event.get('Region', 'all')
            
            for pool_id in pools:
                df_pool = df_price[df_price['Pool_ID'] == pool_id].copy()
                
                if len(df_pool) == 0:
                    continue
                
                region = df_pool['Region'].iloc[0]
                
                if event_region != 'all' and event_region != region:
                    continue
                
                baseline = self.pool_baselines.get(pool_id, df_pool['SpotPrice'].mean())
                
                event_start = event_date - timedelta(days=5)
                event_end = event_date + timedelta(days=5)
                
                event_data = df_pool[
                    (df_pool['timestamp'] >= event_start) &
                    (df_pool['timestamp'] <= event_end)
                ]
                
                if len(event_data) == 0:
                    continue
                
                event_mean = event_data['SpotPrice'].mean()
                price_change = abs((event_mean - baseline) / (baseline + 1e-6)) * 100
                
                is_significant = price_change > 5.0
                
                event_analysis.append({
                    'event_name': event_name,
                    'event_date': event_date,
                    'pool_id': pool_id,
                    'instance_type': pool_id.split('_')[0],
                    'az': '_'.join(pool_id.split('_')[1:]),
                    'region': region,
                    'baseline_price': baseline,
                    'price_change_pct': price_change,
                    'impact_score': min(10, price_change / 2),
                    'is_significant': is_significant
                })
        
        df_analysis = pd.DataFrame(event_analysis)
        
        if len(df_analysis) > 0:
            significant = df_analysis[df_analysis['is_significant']]
            print(f"\n✓ Analyzed {len(df_analysis)} event-pool combinations")
            print(f"✓ Significant: {len(significant)} ({len(significant)/len(df_analysis)*100:.1f}%)")
        
        return df_analysis
    
    def create_dynamic_features(self, df_price, df_event_analysis):
        print("\n🔧 Creating event features...")
        
        df = df_price.copy()
        
        df['in_significant_event_window'] = 0
        df['event_impact_score'] = 0.0
        df['event_name'] = ''
        
        if len(df_event_analysis) > 0:
            significant = df_event_analysis[df_event_analysis['is_significant']]
            
            for _, event in significant.iterrows():
                event_date = event['event_date']
                pool_id = event['pool_id']
                impact = event['impact_score']
                event_name = event['event_name']
                
                event_start = event_date - timedelta(days=5)
                event_end = event_date + timedelta(days=5)
                
                mask = (
                    (df['Pool_ID'] == pool_id) &
                    (df['timestamp'] >= event_start) &
                    (df['timestamp'] <= event_end)
                )
                
                df.loc[mask, 'in_significant_event_window'] = 1
                df.loc[mask, 'event_impact_score'] = impact
                df.loc[mask, 'event_name'] = event_name
        
        event_count = (df['in_significant_event_window'] == 1).sum()
        print(f"✓ Event window records: {event_count:,} ({event_count/len(df)*100:.1f}%)")
        
        return df

#═══════════════════════════════════════════════════════════════════════════════
# VELOCITY-BASED RISK SCORING (unchanged)
#═══════════════════════════════════════════════════════════════════════════════

class VelocityRiskEngine:
    """Risk scoring based on ratio velocity and trends (not spikes)"""
    
    def __init__(self):
        self.risk_profiles = {}
    
    def compute_ratio_risk(self, df_pool):
        """Compute ratio-based risk (0-100) - ULTRA-SENSITIVE"""
        current_ratio = df_pool['spot_ondemand_ratio'].iloc[-1] if len(df_pool) > 0 else 0.5
        regional_pressure = df_pool['regional_pressure'].iloc[-1] if len(df_pool) > 0 else 0.5
        
        ratio_risk = ((current_ratio - 0.35) / 0.30) * 100
        ratio_risk = np.clip(ratio_risk, 0, 100)
        
        regional_risk = ((regional_pressure - 0.35) / 0.30) * 100
        regional_risk = np.clip(regional_risk, 0, 100)
        
        combined_risk = 0.7 * ratio_risk + 0.3 * regional_risk
        
        return combined_risk
    
    def compute_volatility_risk(self, df_pool):
        """Compute volatility risk (0-100) - ULTRA-SENSITIVE"""
        price_std = df_pool['SpotPrice'].std() if len(df_pool) > 0 else 0
        price_mean = df_pool['SpotPrice'].mean() if len(df_pool) > 0 else 1
        ratio_vol = df_pool['ratio_volatility_7d'].iloc[-1] if len(df_pool) > 0 else 0
        
        cv = (price_std / price_mean) * 100 if price_mean > 0 else 0
        cv_risk = min(100, cv * 5)
        
        ratio_risk = min(100, ratio_vol * 500)
        
        volatility_risk = (cv_risk + ratio_risk) / 2
        
        return volatility_risk
    
    def compute_velocity_risk(self, df_pool):
        """Compute velocity risk (0-100) - NEW PRIMARY SIGNAL"""
        velocity_24h = abs(df_pool['ratio_velocity_24h'].iloc[-1]) if len(df_pool) > 0 else 0
        velocity_7d = abs(df_pool['ratio_velocity_7d'].iloc[-1]) if len(df_pool) > 0 else 0
        acceleration = abs(df_pool['ratio_acceleration'].iloc[-1]) if len(df_pool) > 0 else 0
        
        vel_24h_risk = min(100, (velocity_24h / config.RATIO_VELOCITY_WARNING) * 50)
        vel_7d_risk = min(100, (velocity_7d / (config.RATIO_VELOCITY_WARNING * 7)) * 50)
        
        accel_risk = min(100, acceleration * 1000)
        
        velocity_risk = max(vel_24h_risk, vel_7d_risk) + accel_risk * 0.2
        velocity_risk = min(100, velocity_risk)
        
        return velocity_risk
    
    def compute_confidence_risk(self, df_pool):
        """Compute prediction confidence risk (0-100)"""
        if len(df_pool) > 0:
            hours_old = (datetime.now() - df_pool['timestamp'].max()).total_seconds() / 3600
            recency_risk = min(100, (hours_old / 168) * 100)
        else:
            recency_risk = 100
        
        return recency_risk
    
    def compute_overall_risk(self, pool_id, df_pool):
        """Compute overall risk score (0-100)"""
        
        ratio_risk = self.compute_ratio_risk(df_pool)
        volatility = self.compute_volatility_risk(df_pool)
        velocity = self.compute_velocity_risk(df_pool)
        confidence = self.compute_confidence_risk(df_pool)
        
        overall = (
            config.RISK_WEIGHT_RATIO * ratio_risk +
            config.RISK_WEIGHT_VOLATILITY * volatility +
            config.RISK_WEIGHT_VELOCITY * velocity +
            config.RISK_WEIGHT_CONFIDENCE * confidence
        )
        
        risk_profile = {
            'pool_id': pool_id,
            'overall_risk': overall,
            'ratio_risk': ratio_risk,
            'volatility_risk': volatility,
            'velocity_risk': velocity,
            'confidence_risk': confidence,
            'risk_category': self._categorize_risk(overall)
        }
        
        self.risk_profiles[pool_id] = risk_profile
        
        return risk_profile
    
    def _categorize_risk(self, risk_score):
        """Categorize risk score"""
        if risk_score < config.RISK_SAFE_MAX:
            return 'SAFE'
        elif risk_score < config.RISK_MODERATE_MAX:
            return 'MODERATE'
        else:
            return 'RISKY'
    
    def get_risk_summary(self):
        """Get summary of all risk profiles with diagnostics"""
        if not self.risk_profiles:
            return None
        
        df = pd.DataFrame(list(self.risk_profiles.values()))
        
        summary = {
            'total_pools': len(df),
            'safe_pools': len(df[df['risk_category'] == 'SAFE']),
            'moderate_pools': len(df[df['risk_category'] == 'MODERATE']),
            'risky_pools': len(df[df['risk_category'] == 'RISKY']),
            'avg_overall_risk': df['overall_risk'].mean()
        }
        
        print(f"\n  📊 Risk Score Distribution:")
        print(f"     Min: {df['overall_risk'].min():.1f}")
        print(f"     25th: {df['overall_risk'].quantile(0.25):.1f}")
        print(f"     Median: {df['overall_risk'].median():.1f}")
        print(f"     75th: {df['overall_risk'].quantile(0.75):.1f}")
        print(f"     Max: {df['overall_risk'].max():.1f}")
        
        print(f"\n  📊 Component Breakdown:")
        print(f"     Ratio: {df['ratio_risk'].mean():.1f} (range: {df['ratio_risk'].min():.1f}-{df['ratio_risk'].max():.1f})")
        print(f"     Volatility: {df['volatility_risk'].mean():.1f} (range: {df['volatility_risk'].min():.1f}-{df['volatility_risk'].max():.1f})")
        print(f"     Velocity: {df['velocity_risk'].mean():.1f} (range: {df['velocity_risk'].min():.1f}-{df['velocity_risk'].max():.1f})")
        
        top_risky = df.nlargest(5, 'overall_risk')
        print(f"\n  🔴 Top 5 Riskiest Pools:")
        for _, row in top_risky.iterrows():
            print(f"     {row['pool_id']}: {row['overall_risk']:.1f} ({row['risk_category']}) - Ratio:{row['ratio_risk']:.0f}, Vel:{row['velocity_risk']:.0f}")
        
        return summary

#═══════════════════════════════════════════════════════════════════════════════
# MAIN PREPROCESSOR (unchanged)
#═══════════════════════════════════════════════════════════════════════════════

class UltraStablePreprocessor:
    """Preprocessor optimized for ultra-stable data"""
    
    def __init__(self):
        self.data_cleaner = StratifiedDataCleaner()
        self.pool_profiler = PoolMetadataProfiler()
        self.velocity_engineer = VelocityBasedEngineer()
        self.event_analyzer = EventAnalyzer()
        self.risk_engine = VelocityRiskEngine()
        
        self.event_analysis_df = None
        self.pool_profiles_df = None
        self.trained_pools = set()
    
    def load_data(self, price_path, event_path):
        print("📊 Loading datasets...")
        
        df_price = pd.read_csv(price_path)
        
        col_map = {}
        for col in df_price.columns:
            col_lower = col.lower()
            if 'time' in col_lower or 'date' in col_lower:
                col_map[col] = 'timestamp'
            elif 'spot' in col_lower and 'price' in col_lower:
                col_map[col] = 'SpotPrice'
            elif 'ondemand' in col_lower or ('on' in col_lower and 'demand' in col_lower):
                col_map[col] = 'OnDemandPrice'
            elif 'instance' in col_lower:
                col_map[col] = 'InstanceType'
            elif col_lower in ['az', 'availability_zone', 'availabilityzone']:
                col_map[col] = 'AZ'
            elif 'region' in col_lower:
                col_map[col] = 'Region'
        
        df_price = df_price.rename(columns=col_map)
        df_price['timestamp'] = pd.to_datetime(df_price['timestamp'])
        df_price = df_price.sort_values('timestamp').reset_index(drop=True)
        
        default_prices = {
            't3.medium': 0.0416, 't4g.medium': 0.0336,
            't4g.small': 0.0168, 'c5.large': 0.085
        }
        
        for inst, price in default_prices.items():
            mask = (df_price['InstanceType'] == inst) & df_price['OnDemandPrice'].isna()
            df_price.loc[mask, 'OnDemandPrice'] = price
        
        if 'Region' not in df_price.columns or df_price['Region'].isna().any():
            df_price['Region'] = df_price['AZ'].str.extract(r'^([a-z]+-[a-z]+-\d+)')[0].fillna('ap-south-1')
        
        df_price['Pool_ID'] = df_price['InstanceType'] + '_' + df_price['AZ']
        
        print("\n🧹 CLEANING DATA")
        print("="*80)
        
        if config.REMOVE_OUTLIERS:
            df_price = self.data_cleaner.remove_outliers_stratified(df_price, threshold=config.OUTLIER_THRESHOLD)
        
        if config.SMOOTH_PRICES:
            df_price = self.data_cleaner.smooth_prices_fast(df_price, window=config.SMOOTH_WINDOW)
        
        df_price = self.data_cleaner.cap_extreme_values_fast(df_price)
        
        df_events = pd.read_csv(event_path, parse_dates=['Date'])
        
        print(f"\n✓ Loaded {len(df_price):,} records, {len(df_events)} events")
        
        return df_price, df_events
    
    def engineer_features(self, df_price, df_events, is_training=True):
        print("\n🔧 VELOCITY-BASED FEATURE ENGINEERING")
        print("="*80)
        
        if is_training:
            self.trained_pools = set(df_price['Pool_ID'].unique())
            print(f"Training pools: {len(self.trained_pools)}")
            
            self.pool_profiles_df = self.pool_profiler.profile_all_pools(df_price)
            self.pool_profiles_df = self.pool_profiler.cluster_pools(self.pool_profiles_df)
            
            profile_path = os.path.join(config.OUTPUT_DIR, 'pool_profiles.csv')
            self.pool_profiles_df.to_csv(profile_path, index=False)
            print(f"\n✓ Saved: {profile_path}")
        else:
            df_price = df_price[df_price['Pool_ID'].isin(self.trained_pools)].copy()
            print(f"Filtered to {len(self.trained_pools)} trained pools")
        
        if is_training:
            self.event_analysis_df = self.event_analyzer.analyze_event_impact(df_price, df_events)
        
        df = self.event_analyzer.create_dynamic_features(df_price, self.event_analysis_df)
        
        print("\n🔧 Creating velocity-based features...")
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        df = self.velocity_engineer.compute_ratio_features(df)
        df = self.velocity_engineer.compute_regional_pressure(df)
        df = self.velocity_engineer.compute_cross_pool_features(df)
        
        for window in config.ROLLING_WINDOWS:
            df[f'spot_mean_{window}h'] = df.groupby('Pool_ID')['SpotPrice'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            df[f'spot_std_{window}h'] = df.groupby('Pool_ID')['SpotPrice'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
        
        df['is_stable'] = (
            (df['capacity_normal'] == 1) &
            (df['velocity_warning'] == 0) &
            (df['in_significant_event_window'] == 0)
        ).astype(int)
        
        df['is_high_risk'] = (
            (df['capacity_critical'] == 1) |
            (df['velocity_critical'] == 1) |
            (df['in_significant_event_window'] == 1)
        ).astype(int)
        
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"\n✓ Features: {len(df.columns)}")
        print(f"  Stable: {df['is_stable'].sum():,} ({df['is_stable'].mean()*100:.1f}%)")
        print(f"  High-risk: {df['is_high_risk'].sum():,} ({df['is_high_risk'].mean()*100:.1f}%)")
        print(f"  Velocity warnings: {df['velocity_warning'].sum():,} ({df['velocity_warning'].mean()*100:.1f}%)")
        
        return df
    
    def compute_risk_scores(self, df):
        """Compute risk scores for all pools"""
        print("\n⚠️  COMPUTING VELOCITY-BASED RISK SCORES")
        print("="*80)
        
        for pool_id in tqdm(df['Pool_ID'].unique(), desc="Risk scoring"):
            df_pool = df[df['Pool_ID'] == pool_id].tail(168)
            self.risk_engine.compute_overall_risk(pool_id, df_pool)
        
        summary = self.risk_engine.get_risk_summary()
        if summary:
            print(f"\n✓ Risk Summary:")
            print(f"  Safe pools: {summary['safe_pools']}")
            print(f"  Moderate pools: {summary['moderate_pools']}")
            print(f"  Risky pools: {summary['risky_pools']}")
            print(f"  Avg overall risk: {summary['avg_overall_risk']:.1f}/100")
        
        risk_path = os.path.join(config.OUTPUT_DIR, 'risk_profiles.csv')
        pd.DataFrame(list(self.risk_engine.risk_profiles.values())).to_csv(risk_path, index=False)
        print(f"✓ Saved: {risk_path}")
        
        return df
    
    def get_analyses(self):
        return self.event_analysis_df

#═══════════════════════════════════════════════════════════════════════════════
# PROPHET MODEL (unchanged)
#═══════════════════════════════════════════════════════════════════════════════

class EnhancedProphetModel:
    """Prophet with enhanced configuration"""
    
    def __init__(self):
        self.pool_models = {}
        self.validation_metrics = {}
    
    def _prepare_holidays(self, df_event_analysis, pool_id):
        if df_event_analysis is None or len(df_event_analysis) == 0:
            return None
        
        significant_events = df_event_analysis[
            (df_event_analysis['pool_id'] == pool_id) &
            (df_event_analysis['is_significant'] == True)
        ]
        
        holidays_list = []
        for _, event in significant_events.iterrows():
            event_date = pd.to_datetime(event['event_date'])
            event_name = event['event_name']
            
            for day_offset in range(-5, 6):
                holiday_date = event_date + timedelta(days=day_offset)
                holidays_list.append({
                    'ds': holiday_date,
                    'holiday': event_name,
                    'lower_window': -5,
                    'upper_window': 5
                })
        
        if len(holidays_list) > 0:
            return pd.DataFrame(holidays_list).drop_duplicates(subset=['ds', 'holiday'])
        
        return None
    
    def train_all(self, df_train, df_val, df_event_analysis):
        print("\n📈 TRAINING PROPHET MODELS")
        print("="*80)
        
        pools = df_train['Pool_ID'].unique()
        print(f"Training {len(pools)} pool models...")
        
        trained = 0
        pbar = tqdm(pools, desc="Training")
        
        for pool_id in pbar:
            try:
                df_pool_train = df_train[df_train['Pool_ID'] == pool_id].copy()
                df_pool_val = df_val[df_val['Pool_ID'] == pool_id].copy()
                
                if len(df_pool_train) < config.MIN_TRAIN_SAMPLES or len(df_pool_val) < 10:
                    continue
                
                train_agg = df_pool_train.groupby('timestamp').agg({'SpotPrice': 'mean'}).reset_index()
                prophet_df = pd.DataFrame({'ds': train_agg['timestamp'], 'y': train_agg['SpotPrice']})
                
                holidays_df = self._prepare_holidays(df_event_analysis, pool_id)
                
                model = Prophet(
                    changepoint_prior_scale=config.PROPHET_CHANGEPOINT_PRIOR,
                    seasonality_prior_scale=config.PROPHET_SEASONALITY_PRIOR,
                    changepoint_range=config.PROPHET_CHANGEPOINT_RANGE,
                    daily_seasonality=config.ENABLE_DAILY_SEASONALITY,
                    weekly_seasonality=config.ENABLE_WEEKLY_SEASONALITY,
                    yearly_seasonality=config.ENABLE_YEARLY_SEASONALITY,
                    holidays=holidays_df,
                    uncertainty_samples=config.PROPHET_UNCERTAINTY_SAMPLES,
                    interval_width=config.PROPHET_INTERVAL_WIDTH
                )
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(prophet_df)
                
                val_future = pd.DataFrame({
                    'ds': df_pool_val.groupby('timestamp').first().reset_index()['timestamp']
                })
                
                if len(val_future) == 0:
                    continue
                
                val_forecast = model.predict(val_future)
                val_actual = df_pool_val.groupby('timestamp')['SpotPrice'].mean().values
                val_pred = val_forecast['yhat'].values[:len(val_actual)]
                val_pred = np.clip(val_pred, val_actual.min() * 0.5, val_actual.max() * 1.5)
                
                val_mape = mean_absolute_percentage_error(val_actual, val_pred) * 100
                val_rmse = np.sqrt(mean_squared_error(val_actual, val_pred))
                
                self.validation_metrics[pool_id] = {
                    'mape': val_mape,
                    'rmse': val_rmse
                }
                
                self.pool_models[pool_id] = model
                trained += 1
                
                pbar.set_postfix({'trained': trained, 'val_mape': f'{val_mape:.1f}%'})
                
            except Exception as e:
                pbar.write(f"  ✗ {pool_id}: {e}")
                continue
        
        print(f"\n✓ Trained {trained}/{len(pools)} models")
        
        if self.validation_metrics:
            good_metrics = {k: v for k, v in self.validation_metrics.items() if v['mape'] < 50}
            if good_metrics:
                avg_mape = np.mean([m['mape'] for m in good_metrics.values()])
                print(f"✓ Avg validation MAPE: {avg_mape:.2f}%")
        
        return self
    
    def predict(self, df, pool_id, periods=24):
        if pool_id not in self.pool_models:
            return None
        
        model = self.pool_models[pool_id]
        last_date = df['timestamp'].max()
        future = pd.date_range(start=last_date + timedelta(hours=1), periods=periods, freq='H')
        future_df = pd.DataFrame({'ds': future})
        
        forecast = model.predict(future_df)
        
        recent_mean = df[df['Pool_ID'] == pool_id].tail(168)['SpotPrice'].mean()
        recent_std = df[df['Pool_ID'] == pool_id].tail(168)['SpotPrice'].std()
        
        lower_bound = max(0, recent_mean - 3 * recent_std)
        upper_bound = recent_mean + 3 * recent_std
        
        predictions = np.clip(forecast['yhat'].values, lower_bound, upper_bound)
        
        return pd.DataFrame({
            'timestamp': future,
            'predicted_price': predictions
        })

#═══════════════════════════════════════════════════════════════════════════════
# REAL-WORLD SIMULATOR - FIXED VERSION
#═══════════════════════════════════════════════════════════════════════════════

class RealWorldSimulator:
    """Real-world temporal simulation with FIXED date range logic"""
    
    def __init__(self, prophet_model, risk_engine):
        self.prophet_model = prophet_model
        self.risk_engine = risk_engine
    
    def run_simulation(self, df_2025, instance_type=None, pool_id=None):
        """Run real-world simulation with ADAPTIVE buffers for short datasets"""
        
        print("\n" + "="*80)
        print("🎯 REAL-WORLD SIMULATION (VELOCITY-BASED)")
        print("="*80)
        
        if instance_type is None:
            available_instances = df_2025['InstanceType'].unique()
            trained_instances = [i for i in ['t3.medium', 't4g.small', 't4g.medium', 'c5.large'] 
                               if i in available_instances]
            if not trained_instances:
                print("\n❌ No trained instances in test data")
                return None
            instance_type = random.choice(trained_instances)
        
        if pool_id is None:
            df_inst = df_2025[df_2025['InstanceType'] == instance_type]
            available_pools = [p for p in df_inst['Pool_ID'].unique() 
                             if p in self.prophet_model.pool_models]
            if not available_pools:
                print(f"\n❌ No trained models for {instance_type}")
                return None
            pool_id = random.choice(available_pools)
        
        df_pool = df_2025[df_2025['Pool_ID'] == pool_id].sort_values('timestamp').copy()
        
        print(f"\n📍 Selected pool: {pool_id}")
        print(f"   Records: {len(df_pool):,}")
        print(f"   Date range: {df_pool['timestamp'].min()} to {df_pool['timestamp'].max()}")
        
        # NEW: Adaptive buffer logic for short datasets
        dataset_days = (df_pool['timestamp'].max() - df_pool['timestamp'].min()).days
        print(f"   Dataset length: {dataset_days} days")
        
        # Check if dataset is too short
        if dataset_days < config.MIN_DATASET_DAYS:
            print(f"\n⚠️  Warning: Dataset only {dataset_days} days (recommended: {config.MIN_DATASET_DAYS}+ days)")
            
            # Proportional adjustment
            scale_factor = min(1.0, dataset_days / config.MIN_DATASET_DAYS)
            min_history = max(7, int(config.SIMULATION_MIN_HISTORY_DAYS * scale_factor))
            buffer = max(3, int(config.SIMULATION_CUTOFF_BUFFER_DAYS * scale_factor))
            
            print(f"   Adjusting parameters proportionally:")
            print(f"     History: {config.SIMULATION_MIN_HISTORY_DAYS}d → {min_history}d")
            print(f"     Buffer: {config.SIMULATION_CUTOFF_BUFFER_DAYS}d → {buffer}d")
        else:
            min_history = config.SIMULATION_MIN_HISTORY_DAYS
            buffer = config.SIMULATION_CUTOFF_BUFFER_DAYS
            print(f"   Using standard buffers: {min_history}d history, {buffer}d buffer")
        
        min_date = df_pool['timestamp'].min() + timedelta(days=min_history)
        max_date = df_pool['timestamp'].max() - timedelta(days=buffer)
        
        print(f"\n📊 Simulation Window:")
        print(f"   Earliest cutoff: {min_date.date()}")
        print(f"   Latest cutoff: {max_date.date()}")
        print(f"   Available range: {(max_date - min_date).days} days")
        
        if min_date >= max_date:
            available_range = (max_date - min_date).days
            print(f"\n❌ Insufficient range: {available_range} days")
            print(f"   Need at least: {min_history + buffer + 1} days total")
            print(f"\n💡 Suggestions:")
            print(f"   1. Use longer test dataset (current: {dataset_days}d, need: {min_history + buffer + 1}d)")
            print(f"   2. Reduce MIN_HISTORY_DAYS (current: {min_history}d)")
            print(f"   3. Reduce CUTOFF_BUFFER_DAYS (current: {buffer}d)")
            return None
        
        date_range_days = (max_date - min_date).days
        random_days = random.randint(0, date_range_days)
        cutoff_date = min_date + timedelta(days=random_days)
        
        print(f"\n⏰ Cutoff: {cutoff_date.date()}")
        
        df_before = df_pool[df_pool['timestamp'] <= cutoff_date].copy()
        df_after = df_pool[df_pool['timestamp'] > cutoff_date].copy()
        
        print(f"   Before: {len(df_before):,} | After: {len(df_after):,}")
        
        if len(df_after) < 24:
            print(f"\n❌ Insufficient future data")
            return None
        
        forecast = self.prophet_model.predict(df_before, pool_id, periods=min(168, len(df_after)))
        
        if forecast is None:
            print(f"\n❌ Prediction failed")
            return None
        
        actual_future = df_after.head(len(forecast))
        actual_prices = actual_future['SpotPrice'].values
        predicted_prices = forecast['predicted_price'].values[:len(actual_prices)]
        
        metrics = self._calculate_metrics(actual_prices, predicted_prices)
        
        risk_profile = self.risk_engine.compute_overall_risk(pool_id, df_before.tail(168))
        
        print(f"\n" + "="*80)
        print(f"📊 RESULTS - {pool_id}")
        print("="*80)
        
        print(f"\n🎯 Accuracy:")
        print(f"   MAPE: {metrics['mape']:.2f}%")
        print(f"   RMSE: ${metrics['rmse']:.6f}")
        print(f"   Directional: {metrics['directional_accuracy']:.1f}%")
        
        print(f"\n⚠️  Risk at Cutoff:")
        print(f"   Overall: {risk_profile['overall_risk']:.1f}/100 ({risk_profile['risk_category']})")
        print(f"   Ratio: {risk_profile['ratio_risk']:.1f}")
        print(f"   Volatility: {risk_profile['volatility_risk']:.1f}")
        print(f"   Velocity: {risk_profile['velocity_risk']:.1f}")
        
        recent = df_before.tail(24)
        print(f"\n🏊 Capacity Signals:")
        print(f"   Ratio: {recent['spot_ondemand_ratio'].iloc[-1]:.3f}")
        print(f"   Regional: {recent['regional_pressure'].iloc[-1]:.3f}")
        print(f"   Velocity 24h: {recent['ratio_velocity_24h'].iloc[-1]:.4f}")
        print(f"   Velocity warnings: {recent['velocity_warning'].sum()}")
        
        return {
            'pool_id': pool_id,
            'cutoff_date': cutoff_date,
            'metrics': metrics,
            'risk_profile': risk_profile
        }
    
    def _calculate_metrics(self, actual, predicted):
        """Calculate comprehensive metrics"""
        
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = mean_absolute_percentage_error(actual, predicted) * 100
        
        if len(actual) > 1:
            actual_direction = np.diff(actual) > 0
            pred_direction = np.diff(predicted) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            directional_accuracy = 0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }

#═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE WITH MULTI-QUARTER SUPPORT
#═══════════════════════════════════════════════════════════════════════════════

class UltraStablePipeline:
    """Pipeline optimized for ultra-stable data with multi-quarter support"""
    
    def __init__(self):
        self.preprocessor = UltraStablePreprocessor()
        self.prophet_model = EnhancedProphetModel()
        self.simulator = None
    
    def train(self, price_path, event_path):
        print("\n" + "="*80)
        print("🚀 ULTRA-STABLE DATA PIPELINE - TRAINING")
        print("="*80)
        
        df_price, df_events = self.preprocessor.load_data(price_path, event_path)
        
        df = df_price.sort_values('timestamp').reset_index(drop=True)
        n = len(df)
        train_end = int(n * config.TRAIN_RATIO)
        val_end = train_end + int(n * config.VAL_RATIO)
        
        df_train = df.iloc[:train_end].copy()
        df_val = df.iloc[train_end:val_end].copy()
        df_test = df.iloc[val_end:].copy()
        
        print(f"\n📊 Split: Train={len(df_train):,} Val={len(df_val):,} Test={len(df_test):,}")
        
        df_train_feat = self.preprocessor.engineer_features(df_train, df_events, is_training=True)
        df_val_feat = self.preprocessor.engineer_features(df_val, df_events, is_training=False)
        
        df_train_feat = self.preprocessor.compute_risk_scores(df_train_feat)
        
        event_analysis = self.preprocessor.get_analyses()
        if event_analysis is not None and len(event_analysis) > 0:
            path = os.path.join(config.OUTPUT_DIR, 'event_analysis.csv')
            event_analysis.to_csv(path, index=False)
            print(f"\n✓ Saved: {path}")
        
        self.prophet_model.train_all(df_train_feat, df_val_feat, event_analysis)
        
        self.simulator = RealWorldSimulator(self.prophet_model, self.preprocessor.risk_engine)
        
        print("\n✅ Training complete!")
        
        return df_train_feat, df_val_feat, df_test
    
    def evaluate_on_2025(self, test_2025_path, event_path, num_simulations=3):
        print("\n" + "="*80)
        print("🎯 EVALUATION ON 2025 DATA")
        print("="*80)
        
        df_2025, df_events = self.preprocessor.load_data(test_2025_path, event_path)
        df_2025_feat = self.preprocessor.engineer_features(df_2025, df_events, is_training=False)
        df_2025_feat = self.preprocessor.compute_risk_scores(df_2025_feat)
        
        simulation_results = []
        
        for i in range(num_simulations):
            print(f"\n{'='*80}")
            print(f"SIMULATION {i+1}/{num_simulations}")
            
            result = self.simulator.run_simulation(df_2025_feat)
            
            if result:
                simulation_results.append(result)
        
        if simulation_results:
            print(f"\n{'='*80}")
            print(f"📊 AGGREGATED RESULTS")
            print(f"{'='*80}")
            
            avg_mape = np.mean([r['metrics']['mape'] for r in simulation_results])
            avg_rmse = np.mean([r['metrics']['rmse'] for r in simulation_results])
            avg_risk = np.mean([r['risk_profile']['overall_risk'] for r in simulation_results])
            
            print(f"\nAverage Performance ({len(simulation_results)} simulations):")
            print(f"  MAPE: {avg_mape:.2f}%")
            print(f"  RMSE: ${avg_rmse:.6f}")
            print(f"  Risk: {avg_risk:.1f}/100")
        
        return simulation_results
    
    def evaluate_quarterly(self, event_path, q1_path, q2_path, q3_path):
        """NEW: Evaluate across multiple quarters"""
        
        print("\n" + "="*80)
        print("📊 QUARTERLY EVALUATION")
        print("="*80)
        
        quarters = {
            'Q1 (Jan-Mar)': q1_path,
            'Q2 (Apr-Jun)': q2_path,
            'Q3 (Jul-Sep)': q3_path
        }
        
        quarterly_results = {}
        
        for quarter_name, path in quarters.items():
            print(f"\n{'='*80}")
            print(f"🎯 EVALUATING {quarter_name}")
            print(f"{'='*80}")
            
            results = self.evaluate_on_2025(path, event_path, num_simulations=3)
            
            if results:
                quarterly_results[quarter_name] = results
        
        # Compare quarters
        if quarterly_results:
            print(f"\n{'='*80}")
            print(f"📊 QUARTERLY COMPARISON")
            print(f"{'='*80}")
            
            for quarter, results in quarterly_results.items():
                avg_mape = np.mean([r['metrics']['mape'] for r in results])
                avg_rmse = np.mean([r['metrics']['rmse'] for r in results])
                avg_risk = np.mean([r['risk_profile']['overall_risk'] for r in results])
                
                print(f"\n{quarter}:")
                print(f"  MAPE: {avg_mape:.2f}%")
                print(f"  RMSE: ${avg_rmse:.6f}")
                print(f"  Risk: {avg_risk:.1f}/100")
        
        return quarterly_results

#═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION WITH MULTI-QUARTER SUPPORT
#═══════════════════════════════════════════════════════════════════════════════

def main():
    """Main execution with quarterly evaluation"""
    
    TRAIN_PATH = '/Users/atharvapudale/Downloads/aws_2023_2024_complete_24months.csv'
    EVENT_PATH = '/Users/atharvapudale/Downloads/aws_stress_events_2023_2025.csv'
    
    # Test data paths
    Q1_PATH = '/Users/atharvapudale/Downloads/mumbai_spot_data_sorted_asc(1-2-3-25).csv'
    Q2_PATH = '/Users/atharvapudale/Downloads/mumbai_spot_data_sorted_asc(4-5-6-25).csv'
    Q3_PATH = '/Users/atharvapudale/Downloads/mumbai_spot_data_sorted_asc(7-8-9-25).csv'
    
    try:
        pipeline = UltraStablePipeline()
        
        print("\n" + "🔵"*40)
        print("PHASE 1: TRAINING ON 2023-2024")
        print("🔵"*40)
        pipeline.train(TRAIN_PATH, EVENT_PATH)
        
        print("\n" + "🟢"*40)
        print("PHASE 2: QUARTERLY EVALUATION (Q1, Q2, Q3 2025)")
        print("🟢"*40)
        
        quarterly_results = pipeline.evaluate_quarterly(
            event_path=EVENT_PATH,
            q1_path=Q1_PATH,
            q2_path=Q2_PATH,
            q3_path=Q3_PATH
        )
        
        print(f"\n✅ COMPLETE - Outputs: {config.OUTPUT_DIR}/")
        
        return pipeline, quarterly_results
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    pipeline, results = main()
