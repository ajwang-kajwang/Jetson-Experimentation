# baseline/performance_analyzer.py
"""
Performance analysis and insights generation
"""
from typing import Dict, List

class PerformanceAnalyzer:
    """
    Analyzes metrics and provides insights
    No knowledge of collection or display
    """
    
    @staticmethod
    def analyze_metrics(metrics: Dict) -> Dict:
        """
        Analyze metrics and generate insights
        Input: metrics from MetricsCollector
        Output: human-readable insights
        """
        insights = {}
        
        if not metrics:
            return {'error': 'No metrics to analyze'}
        
        # Get key metrics
        mean_lat = metrics.get('mean_latency_ms', 0)
        p95_lat = metrics.get('p95_latency_ms', 0)
        p99_lat = metrics.get('p99_latency_ms', 0)
        fps = metrics.get('fps', 0)
        
        # Stability analysis
        if mean_lat > 0:
            stability_ratio = p95_lat / mean_lat
            insights['stability_ratio'] = stability_ratio
            
            if stability_ratio < 1.2:
                insights['stability'] = "✓ Excellent - very consistent"
            elif stability_ratio < 1.5:
                insights['stability'] = "✓ Good - minor variations"
            else:
                insights['stability'] = "✗ Poor - significant spikes"
        
        # Performance category
        if mean_lat < 16.67:  # 60 FPS
            insights['performance'] = "✓ Excellent - 60+ FPS capable"
        elif mean_lat < 33.33:  # 30 FPS
            insights['performance'] = "✓ Good - Real-time (30+ FPS)"
        elif mean_lat < 50:  # 20 FPS
            insights['performance'] = "⚠ Fair - Interactive (20-30 FPS)"
        else:
            insights['performance'] = "✗ Poor - Consider optimizations"
        
        # Tail latency
        if p95_lat > 0 and p99_lat > 0:
            tail_ratio = p99_lat / p95_lat
            if tail_ratio > 1.5:
                insights['tail_latency'] = "⚠ High tail latency detected"
        
        # Recommendations
        recommendations = []
        if mean_lat > 33:
            recommendations.append("Consider TensorRT optimization")
            recommendations.append("Try smaller model (yolov8s/yolov8n)")
        
        if stability_ratio > 1.5:
            recommendations.append("Check for thermal throttling")
            recommendations.append("Monitor GPU memory usage")
        
        if recommendations:
            insights['recommendations'] = recommendations
        
        return insights
    
    @staticmethod
    def compare_metrics(baseline: Dict, optimized: Dict) -> Dict:
        """
        Compare two sets of metrics (for future tasks)
        """
        comparison = {}
        
        # Calculate improvements
        for key in ['mean_latency_ms', 'p95_latency_ms', 'fps']:
            if key in baseline and key in optimized:
                baseline_val = baseline[key]
                optimized_val = optimized[key]
                
                if key == 'fps':
                    # Higher is better for FPS
                    improvement = ((optimized_val - baseline_val) / baseline_val) * 100
                else:
                    # Lower is better for latency
                    improvement = ((baseline_val - optimized_val) / baseline_val) * 100
                
                comparison[f'{key}_improvement'] = f"{improvement:.1f}%"
        
        return comparison