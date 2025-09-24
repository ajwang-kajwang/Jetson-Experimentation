
# ============================================================================
# PERFORMANCE ANALYZER
# ============================================================================
class PerformanceAnalyzer:
    """Analyzes metrics and provides insights"""
    
    @staticmethod
    def analyze(metrics: Dict[str, float]) -> Dict[str, str]:
        """Analyze metrics and return insights"""
        insights = {}
        
        if not metrics:
            return insights
        
        mean_lat = metrics.get('mean_latency_ms', 0)
        p95_lat = metrics.get('p95_latency_ms', 0)
        p99_lat = metrics.get('p99_latency_ms', 0)
        
        # Stability analysis
        if mean_lat > 0:
            stability_ratio = p95_lat / mean_lat
            insights['stability_ratio'] = f"{stability_ratio:.2f}"
            
            if stability_ratio < 1.2:
                insights['stability'] = "✓ Excellent - consistent performance"
            elif stability_ratio < 1.5:
                insights['stability'] = "⚠ Good - minor variations"
            else:
                insights['stability'] = "✗ Poor - significant spikes"
        
        # Tail latency analysis
        if p95_lat > 0:
            tail_ratio = p99_lat / p95_lat
            insights['tail_ratio'] = f"{tail_ratio:.2f}"
            
            if tail_ratio < 1.2:
                insights['tail_behavior'] = "✓ Minimal extreme outliers"
            else:
                insights['tail_behavior'] = "⚠ Significant tail latency"
        
        # Performance category
        if mean_lat < 20:
            insights['category'] = "✓ Real-time capable (<20ms)"
        elif mean_lat < 33:
            insights['category'] = "✓ Near real-time (30+ FPS)"
        elif mean_lat < 50:
            insights['category'] = "⚠ Interactive (20-30 FPS)"
        else:
            insights['category'] = "✗ Batch processing recommended"
        
        # Recommendations
        recommendations = []
        if mean_lat > 30:
            recommendations.append("Consider model quantization")
            recommendations.append("Try TensorRT optimization")
        if stability_ratio > 1.5:
            recommendations.append("Investigate GPU throttling")
        if metrics.get('jitter_ms', 0) > 5:
            recommendations.append("High jitter - check system load")
        
        if recommendations:
            insights['recommendations'] = " | ".join(recommendations)
        
        return insights
