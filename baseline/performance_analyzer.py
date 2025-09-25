from typing import Dict 

class PerformanceAnalyzer:
    """Optional analyzer for enhanced insights"""
    
    @staticmethod
    def analyze(metrics: Dict) -> Dict:
        """Provide simple analysis"""
        insights = {}
        if 'mean_latency_ms' in metrics and 'p95_latency_ms' in metrics:
            ratio = metrics['p95_latency_ms'] / metrics['mean_latency_ms']
            insights['stability_ratio'] = f"{ratio:.2f}"
            if ratio < 1.5:
                insights['stability'] = "Good"
            else:
                insights['stability'] = "Poor - high variance"
        return insights