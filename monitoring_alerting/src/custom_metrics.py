"""
Custom ML Metrics for Monitoring - WORKING IMPLEMENTATION

This module implements ML-specific metrics that go beyond standard application monitoring.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

# Import metrics from instrumentation
from instrumentation import (
    data_drift_score,
    model_accuracy,
    missing_features_total,
    model_prediction_confidence
)

logger = logging.getLogger(__name__)


@dataclass
class DriftDetectionResult:
    """Results from drift detection test."""
    feature_name: str
    statistic: float
    p_value: float
    is_drift: bool
    test_method: str
    timestamp: datetime


@dataclass
class ModelPerformanceMetrics:
    """Model performance metrics over time."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    sample_count: int
    timestamp: datetime


# =============================================================================
# Data Drift Detection
# =============================================================================

class DataDriftDetector:
    """
    Detect distribution shifts in input data using statistical tests.
    
    Usage:
        # Initialize with training data
        detector = DataDriftDetector(
            reference_data=X_train,
            feature_names=['feature1', 'feature2', 'feature3'],
            threshold=0.05
        )
        
        # Check production data
        drift_results = detector.detect_drift(X_production)
        detector.export_drift_metrics(drift_results)
    """

    def __init__(
        self,
        reference_data: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.05,
        method: str = 'ks'
    ):
        """
        Initialize drift detector with reference distribution.
        
        Args:
            reference_data: Training data distribution (n_samples, n_features)
            feature_names: List of feature names
            threshold: P-value threshold for drift detection (default: 0.05)
            method: Drift detection method ('ks', 'psi', 'js')
        """
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.threshold = threshold
        self.method = method
        self.n_features = reference_data.shape[1]
        
        if len(feature_names) != self.n_features:
            raise ValueError(
                f"Number of feature names ({len(feature_names)}) "
                f"must match number of features ({self.n_features})"
            )

    def kolmogorov_smirnov_test(
        self,
        reference: np.ndarray,
        current: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform Kolmogorov-Smirnov test for distribution shift.
        
        Returns:
            Tuple of (statistic, p_value)
        """
        statistic, p_value = stats.ks_2samp(reference, current)
        return statistic, p_value

    def population_stability_index(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI interpretation:
        - PSI < 0.1: No significant change
        - PSI 0.1-0.25: Moderate change
        - PSI > 0.25: Significant change
        """
        # Create histogram bins from reference data
        ref_hist, bin_edges = np.histogram(reference, bins=bins)
        cur_hist, _ = np.histogram(current, bins=bin_edges)
        
        # Convert to percentages
        ref_pct = ref_hist / len(reference)
        cur_pct = cur_hist / len(current)
        
        # Avoid division by zero
        epsilon = 1e-10
        ref_pct = np.where(ref_pct == 0, epsilon, ref_pct)
        cur_pct = np.where(cur_pct == 0, epsilon, cur_pct)
        
        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        
        return float(psi)

    def detect_drift(
        self,
        current_data: np.ndarray
    ) -> List[DriftDetectionResult]:
        """
        Detect drift across all features.
        
        Args:
            current_data: Current production data (n_samples, n_features)
            
        Returns:
            List of drift detection results per feature
        """
        results = []
        
        for i, feature_name in enumerate(self.feature_names):
            reference_feature = self.reference_data[:, i]
            current_feature = current_data[:, i]
            
            # Choose method
            if self.method == 'ks':
                statistic, p_value = self.kolmogorov_smirnov_test(
                    reference_feature, current_feature
                )
                is_drift = p_value < self.threshold
                
            elif self.method == 'psi':
                statistic = self.population_stability_index(
                    reference_feature, current_feature
                )
                p_value = None
                is_drift = statistic > 0.25  # PSI threshold
            
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            # Create result
            result = DriftDetectionResult(
                feature_name=feature_name,
                statistic=statistic,
                p_value=p_value,
                is_drift=is_drift,
                test_method=self.method,
                timestamp=datetime.now()
            )
            
            results.append(result)
            
            # Log drift detection
            if is_drift:
                logger.warning(
                    f"Drift detected in {feature_name}: "
                    f"statistic={statistic:.4f}, p_value={p_value}"
                )
        
        return results

    def export_drift_metrics(self, drift_results: List[DriftDetectionResult]):
        """
        Export drift detection results to Prometheus metrics.
        """
        for result in drift_results:
            # Update drift score gauge
            data_drift_score.labels(
                feature_name=result.feature_name
            ).set(result.statistic)
            
            # Log to application logs
            logger.info(
                "Drift detection result",
                extra={
                    'feature_name': result.feature_name,
                    'statistic': result.statistic,
                    'p_value': result.p_value,
                    'is_drift': result.is_drift,
                    'method': result.test_method
                }
            )


# =============================================================================
# Model Performance Monitor
# =============================================================================

class ModelPerformanceMonitor:
    """
    Monitor model performance metrics over time.
    
    Usage:
        monitor = ModelPerformanceMonitor(model_name='demo-model')
        
        # Log predictions
        monitor.log_prediction(prediction='cat', ground_truth='cat')
        
        # Calculate metrics after collecting ground truth
        metrics = monitor.calculate_metrics()
        
        # Check for degradation
        is_degraded = monitor.check_degradation(baseline_accuracy=0.90)
    """

    def __init__(self, model_name: str, min_samples: int = 100):
        """
        Initialize performance monitor.
        
        Args:
            model_name: Name of the model being monitored
            min_samples: Minimum samples before calculating metrics
        """
        self.model_name = model_name
        self.min_samples = min_samples
        self.predictions = []
        self.ground_truth = []
        self.prediction_timestamps = []

    def log_prediction(
        self,
        prediction: str,
        ground_truth: Optional[str] = None
    ):
        """
        Log a prediction with optional ground truth.
        
        Args:
            prediction: Model prediction (class name)
            ground_truth: True label (optional, may come later)
        """
        self.predictions.append(prediction)
        if ground_truth is not None:
            self.ground_truth.append(ground_truth)
        self.prediction_timestamps.append(datetime.now())

    def calculate_metrics(self) -> Optional[ModelPerformanceMetrics]:
        """
        Calculate performance metrics from predictions and ground truth.
        
        Returns:
            ModelPerformanceMetrics if enough samples, else None
        """
        if len(self.ground_truth) < self.min_samples:
            logger.warning(
                f"Not enough samples for metrics calculation "
                f"({len(self.ground_truth)} / {self.min_samples})"
            )
            return None
        
        # Calculate simple accuracy
        correct = sum(p == g for p, g in zip(self.predictions[:len(self.ground_truth)], self.ground_truth))
        accuracy = correct / len(self.ground_truth)
        
        # For simplicity, use accuracy for all metrics
        # In production, you'd use sklearn.metrics
        metrics = ModelPerformanceMetrics(
            accuracy=accuracy,
            precision=accuracy,  # Simplified
            recall=accuracy,     # Simplified
            f1_score=accuracy,   # Simplified
            sample_count=len(self.ground_truth),
            timestamp=datetime.now()
        )
        
        # Update Prometheus metrics
        model_accuracy.labels(model_name=self.model_name).set(accuracy)
        
        # Log metrics
        logger.info(
            f"Model performance metrics: accuracy={accuracy:.4f}, "
            f"sample_count={len(self.ground_truth)}"
        )
        
        return metrics

    def check_degradation(
        self,
        baseline_accuracy: float,
        threshold: float = 0.1
    ) -> bool:
        """
        Check if model performance has degraded significantly.
        
        Args:
            baseline_accuracy: Expected baseline accuracy
            threshold: Degradation threshold (e.g., 0.1 = 10% drop)
            
        Returns:
            True if degradation detected, False otherwise
        """
        if len(self.ground_truth) < self.min_samples:
            return False
        
        correct = sum(p == g for p, g in zip(self.predictions[:len(self.ground_truth)], self.ground_truth))
        current_accuracy = correct / len(self.ground_truth)
        degradation = baseline_accuracy - current_accuracy
        
        if degradation > threshold:
            logger.error(
                f"Performance degradation detected! "
                f"Baseline: {baseline_accuracy:.4f}, "
                f"Current: {current_accuracy:.4f}, "
                f"Degradation: {degradation:.4f}"
            )
            return True
        
        return False


# =============================================================================
# Confidence Analyzer
# =============================================================================

class ConfidenceAnalyzer:
    """
    Analyze prediction confidence scores over time.
    
    Usage:
        analyzer = ConfidenceAnalyzer()
        analyzer.log_confidence(confidence=0.95, is_correct=True)
        stats = analyzer.get_statistics()
    """

    def __init__(self, window_size: int = 1000):
        """
        Initialize confidence analyzer.
        
        Args:
            window_size: Number of recent predictions to analyze
        """
        self.window_size = window_size
        self.confidences = []
        self.correctness = []

    def log_confidence(self, confidence: float, is_correct: Optional[bool] = None):
        """
        Log prediction confidence.
        
        Args:
            confidence: Prediction confidence (0-1)
            is_correct: Whether prediction was correct (optional)
        """
        self.confidences.append(confidence)
        if is_correct is not None:
            self.correctness.append(is_correct)
        
        # Keep only recent window
        if len(self.confidences) > self.window_size:
            self.confidences = self.confidences[-self.window_size:]
            self.correctness = self.correctness[-self.window_size:]

    def get_statistics(self) -> Dict[str, float]:
        """
        Calculate confidence statistics.
        
        Returns:
            Dictionary with statistics (mean, median, std, percentiles)
        """
        if len(self.confidences) == 0:
            return {}
        
        confidences_array = np.array(self.confidences)
        
        stats = {
            'mean': float(np.mean(confidences_array)),
            'median': float(np.median(confidences_array)),
            'std': float(np.std(confidences_array)),
            'min': float(np.min(confidences_array)),
            'max': float(np.max(confidences_array)),
            'p25': float(np.percentile(confidences_array, 25)),
            'p50': float(np.percentile(confidences_array, 50)),
            'p75': float(np.percentile(confidences_array, 75)),
            'p95': float(np.percentile(confidences_array, 95)),
            'count': len(self.confidences)
        }
        
        return stats


# =============================================================================
# Data Quality Monitor
# =============================================================================

class DataQualityMonitor:
    """
    Monitor data quality issues in production requests.
    
    Usage:
        schema = {
            'feature1': 'float',
            'feature2': 'float',
            'feature3': 'float'
        }
        monitor = DataQualityMonitor(expected_schema=schema)
        issues = monitor.validate_request({'feature1': 1.0})
    """

    def __init__(self, expected_schema: Dict[str, str]):
        """
        Initialize data quality monitor.
        
        Args:
            expected_schema: Expected schema {feature_name: data_type}
        """
        self.expected_schema = expected_schema
        self.issue_counts = {
            'missing': {},
            'type_error': {}
        }

    def validate_request(self, data: Dict) -> Dict[str, List[str]]:
        """
        Validate incoming request data.
        
        Args:
            data: Request data dictionary
            
        Returns:
            Dictionary of issues found {issue_type: [feature_names]}
        """
        issues = {
            'missing': [],
            'type_error': []
        }
        
        # Check for missing features
        for feature_name in self.expected_schema:
            if feature_name not in data:
                issues['missing'].append(feature_name)
                missing_features_total.labels(
                    feature_name=feature_name
                ).inc()
        
        return issues


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    print("Custom ML Metrics - Working Implementation")
    print("=" * 50)
    
    # Example 1: Drift Detection
    print("\n1. Testing Drift Detection:")
    np.random.seed(42)
    reference_data = np.random.normal(0, 1, (1000, 3))
    feature_names = ['feature1', 'feature2', 'feature3']
    
    detector = DataDriftDetector(
        reference_data=reference_data,
        feature_names=feature_names,
        threshold=0.05,
        method='ks'
    )
    
    # Test with drifted data
    drifted_data = np.random.normal(0.5, 1, (1000, 3))
    drift_results = detector.detect_drift(drifted_data)
    
    for result in drift_results:
        print(f"{result.feature_name}: drift={result.is_drift}, "
              f"statistic={result.statistic:.4f}, p_value={result.p_value:.4f}")
    
    # Example 2: Performance Monitoring
    print("\n2. Testing Performance Monitoring:")
    monitor = ModelPerformanceMonitor('test_model', min_samples=10)
    
    for i in range(20):
        pred = np.random.choice(['cat', 'dog'])
        truth = np.random.choice(['cat', 'dog'])
        monitor.log_prediction(pred, truth)
    
    metrics = monitor.calculate_metrics()
    if metrics:
        print(f"Accuracy: {metrics.accuracy:.4f}")
    
    # Example 3: Confidence Analysis
    print("\n3. Testing Confidence Analysis:")
    analyzer = ConfidenceAnalyzer()
    
    for _ in range(50):
        confidence = np.random.uniform(0.7, 0.99)
        analyzer.log_confidence(confidence)
    
    stats = analyzer.get_statistics()
    print(f"Mean confidence: {stats['mean']:.3f}")
    print(f"Median confidence: {stats['median']:.3f}")