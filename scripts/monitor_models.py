"""
Model monitoring script for tracking performance and detecting drift.

This module provides utilities for:
- Performance monitoring
- Data drift detection
- Model drift detection
- Alerting and notifications
"""

import os
import sys
import json
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from collections import defaultdict
import mlflow
from mlflow.tracking import MlflowClient


class PerformanceMonitor:
    """Monitors model performance over time."""
    
    def __init__(self, model_name: str, tracking_uri: Optional[str] = None):
        """
        Initialize performance monitor.
        
        Args:
            model_name: Name of the model to monitor
            tracking_uri: MLflow tracking URI
        """
        self.model_name = model_name
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.client = MlflowClient()
        self.metrics_history = []
    
    def collect_metrics(
        self,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None
    ):
        """
        Collect metrics for monitoring.
        
        Args:
            metrics: Dictionary of metric names and values
            timestamp: Timestamp for metrics (default: now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        metric_entry = {
            'timestamp': timestamp.isoformat(),
            'metrics': metrics
        }
        
        self.metrics_history.append(metric_entry)
    
    def get_metric_statistics(
        self,
        metric_name: str,
        window_days: int = 7
    ) -> Dict[str, float]:
        """
        Get statistics for a metric over time window.
        
        Args:
            metric_name: Name of the metric
            window_days: Time window in days
            
        Returns:
            Dictionary with mean, std, min, max
        """
        cutoff_date = datetime.now() - timedelta(days=window_days)
        
        values = []
        for entry in self.metrics_history:
            entry_time = datetime.fromisoformat(entry['timestamp'])
            if entry_time >= cutoff_date:
                if metric_name in entry['metrics']:
                    values.append(entry['metrics'][metric_name])
        
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }
    
    def detect_performance_degradation(
        self,
        metric_name: str,
        threshold: float = 0.1,
        window_days: int = 7
    ) -> bool:
        """
        Detect if performance has degraded.
        
        Args:
            metric_name: Name of the metric
            threshold: Degradation threshold (e.g., 0.1 for 10% drop)
            window_days: Time window for comparison
            
        Returns:
            True if degradation detected
        """
        stats = self.get_metric_statistics(metric_name, window_days)
        
        if not stats:
            return False
        
        # Get baseline from older data
        older_cutoff = datetime.now() - timedelta(days=window_days * 2)
        baseline_cutoff = datetime.now() - timedelta(days=window_days)
        
        baseline_values = []
        for entry in self.metrics_history:
            entry_time = datetime.fromisoformat(entry['timestamp'])
            if older_cutoff <= entry_time < baseline_cutoff:
                if metric_name in entry['metrics']:
                    baseline_values.append(entry['metrics'][metric_name])
        
        if not baseline_values:
            return False
        
        baseline_mean = np.mean(baseline_values)
        current_mean = stats['mean']
        
        # Check for degradation (lower is worse for most metrics)
        degradation = (baseline_mean - current_mean) / baseline_mean
        
        return degradation > threshold
    
    def save_metrics(self, filepath: str):
        """Save metrics history to file."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        print(f"✓ Metrics saved to {filepath}")
    
    def load_metrics(self, filepath: str):
        """Load metrics history from file."""
        with open(filepath, 'r') as f:
            self.metrics_history = json.load(f)
        
        print(f"✓ Metrics loaded from {filepath}")


class DriftDetector:
    """Detects data and model drift."""
    
    def __init__(self):
        """Initialize drift detector."""
        self.reference_data = None
        self.feature_stats = {}
    
    def fit_reference(self, data: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Fit on reference data.
        
        Args:
            data: Reference data array
            feature_names: Names of features
        """
        self.reference_data = data
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(data.shape[1])]
        
        # Calculate statistics for each feature
        for i, name in enumerate(feature_names):
            feature_data = data[:, i]
            self.feature_stats[name] = {
                'mean': np.mean(feature_data),
                'std': np.std(feature_data),
                'min': np.min(feature_data),
                'max': np.max(feature_data),
                'q25': np.percentile(feature_data, 25),
                'q50': np.percentile(feature_data, 50),
                'q75': np.percentile(feature_data, 75)
            }
    
    def detect_drift(
        self,
        data: np.ndarray,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.1
    ) -> Dict[str, bool]:
        """
        Detect drift in new data.
        
        Args:
            data: New data array
            feature_names: Names of features
            threshold: Drift threshold (e.g., 0.1 for 10% change)
            
        Returns:
            Dictionary mapping feature names to drift status
        """
        if self.reference_data is None:
            raise ValueError("Must fit reference data first")
        
        if feature_names is None:
            feature_names = list(self.feature_stats.keys())
        
        drift_status = {}
        
        for i, name in enumerate(feature_names):
            if name not in self.feature_stats:
                continue
            
            feature_data = data[:, i]
            ref_stats = self.feature_stats[name]
            
            # Calculate current statistics
            current_mean = np.mean(feature_data)
            current_std = np.std(feature_data)
            
            # Detect drift using mean and std changes
            mean_change = abs(current_mean - ref_stats['mean']) / (ref_stats['std'] + 1e-10)
            std_change = abs(current_std - ref_stats['std']) / (ref_stats['std'] + 1e-10)
            
            drift_detected = mean_change > threshold or std_change > threshold
            drift_status[name] = drift_detected
        
        return drift_status
    
    def calculate_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            expected: Expected (reference) distribution
            actual: Actual (current) distribution
            bins: Number of bins for discretization
            
        Returns:
            PSI value
        """
        # Create bins
        breakpoints = np.linspace(
            min(expected.min(), actual.min()),
            max(expected.max(), actual.max()),
            bins + 1
        )
        
        # Calculate distributions
        expected_hist, _ = np.histogram(expected, bins=breakpoints)
        actual_hist, _ = np.histogram(actual, bins=breakpoints)
        
        # Normalize
        expected_pct = expected_hist / len(expected)
        actual_pct = actual_hist / len(actual)
        
        # Avoid division by zero
        expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
        actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
        
        # Calculate PSI
        psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
        
        return float(psi)


class ModelMonitor:
    """Comprehensive model monitoring."""
    
    def __init__(
        self,
        model_name: str,
        config_path: Optional[str] = None,
        tracking_uri: Optional[str] = None
    ):
        """
        Initialize model monitor.
        
        Args:
            model_name: Name of the model
            config_path: Path to monitoring configuration
            tracking_uri: MLflow tracking URI
        """
        self.model_name = model_name
        self.performance_monitor = PerformanceMonitor(model_name, tracking_uri)
        self.drift_detector = DriftDetector()
        
        self.config = self._load_config(config_path)
        self.alerts = []
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load monitoring configuration."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('monitoring', {})
        
        return {
            'performance_threshold': 0.1,
            'drift_threshold': 0.1,
            'psi_threshold': 0.2,
            'window_days': 7,
            'alert_email': None
        }
    
    def monitor_batch(
        self,
        predictions: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
        features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Monitor a batch of predictions.
        
        Args:
            predictions: Model predictions
            ground_truth: True labels (if available)
            features: Input features
            
        Returns:
            Monitoring report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'batch_size': len(predictions),
            'alerts': []
        }
        
        # Calculate performance metrics if ground truth available
        if ground_truth is not None:
            metrics = self._calculate_metrics(predictions, ground_truth)
            report['metrics'] = metrics
            
            # Collect metrics for monitoring
            self.performance_monitor.collect_metrics(metrics)
            
            # Check for performance degradation
            for metric_name, value in metrics.items():
                degraded = self.performance_monitor.detect_performance_degradation(
                    metric_name,
                    threshold=self.config['performance_threshold']
                )
                
                if degraded:
                    alert = f"Performance degradation detected in {metric_name}"
                    report['alerts'].append(alert)
                    self.alerts.append({
                        'timestamp': datetime.now().isoformat(),
                        'type': 'performance',
                        'message': alert
                    })
        
        # Check for data drift if features available
        if features is not None and self.drift_detector.reference_data is not None:
            drift_status = self.drift_detector.detect_drift(
                features,
                threshold=self.config['drift_threshold']
            )
            
            report['drift'] = drift_status
            
            # Check for drift alerts
            drifted_features = [k for k, v in drift_status.items() if v]
            if drifted_features:
                alert = f"Data drift detected in features: {', '.join(drifted_features)}"
                report['alerts'].append(alert)
                self.alerts.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'drift',
                    'message': alert
                })
        
        return report
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        ground_truth: np.ndarray
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        metrics = {}
        
        try:
            metrics['accuracy'] = accuracy_score(ground_truth, predictions)
            metrics['f1'] = f1_score(ground_truth, predictions, average='weighted', zero_division=0)
            metrics['precision'] = precision_score(ground_truth, predictions, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(ground_truth, predictions, average='weighted', zero_division=0)
        except Exception as e:
            print(f"Warning: Could not calculate metrics: {e}")
        
        return metrics
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate monitoring report.
        
        Args:
            output_path: Path to save report
            
        Returns:
            Report as string
        """
        report = {
            'model': self.model_name,
            'generated_at': datetime.now().isoformat(),
            'metrics_history_size': len(self.performance_monitor.metrics_history),
            'total_alerts': len(self.alerts),
            'recent_alerts': self.alerts[-10:] if self.alerts else []
        }
        
        # Get metric statistics
        if self.performance_monitor.metrics_history:
            latest_metrics = self.performance_monitor.metrics_history[-1]['metrics']
            report['latest_metrics'] = latest_metrics
            
            # Get statistics for each metric
            report['metric_statistics'] = {}
            for metric_name in latest_metrics.keys():
                stats = self.performance_monitor.get_metric_statistics(
                    metric_name,
                    window_days=self.config['window_days']
                )
                report['metric_statistics'][metric_name] = stats
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"✓ Report saved to {output_path}")
        
        # Print summary
        self._print_report(report)
        
        return json.dumps(report, indent=2)
    
    def _print_report(self, report: Dict):
        """Print monitoring report."""
        print("\n" + "=" * 60)
        print(f"Model Monitoring Report: {report['model']}")
        print("=" * 60)
        print(f"Generated: {report['generated_at']}")
        print(f"Total Alerts: {report['total_alerts']}")
        
        if 'latest_metrics' in report:
            print("\nLatest Metrics:")
            for metric, value in report['latest_metrics'].items():
                print(f"  {metric}: {value:.4f}")
        
        if report['recent_alerts']:
            print("\nRecent Alerts:")
            for alert in report['recent_alerts']:
                print(f"  [{alert['timestamp']}] {alert['type']}: {alert['message']}")
        
        print("=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Monitor model performance and detect drift"
    )
    parser.add_argument(
        '--model-name',
        required=True,
        help='Name of the model to monitor'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/mlops_config.yaml',
        help='Path to monitoring configuration'
    )
    parser.add_argument(
        '--tracking-uri',
        type=str,
        default=None,
        help='MLflow tracking URI'
    )
    parser.add_argument(
        '--report',
        type=str,
        help='Path to save monitoring report'
    )
    
    args = parser.parse_args()
    
    # Initialize monitor
    monitor = ModelMonitor(
        model_name=args.model_name,
        config_path=args.config,
        tracking_uri=args.tracking_uri
    )
    
    # Generate report
    monitor.generate_report(output_path=args.report)
    
    print("\n✓ Monitoring completed")


if __name__ == '__main__':
    main()
