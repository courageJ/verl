import unittest
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from verl.utils.tracking import Tracking

class TestOtelTracking(unittest.TestCase):
    def setUp(self):
        # Setup InMemory Reader to capture metrics
        self.reader = InMemoryMetricReader()
        self.provider = MeterProvider(metric_readers=[self.reader])
        metrics.set_meter_provider(self.provider)

    def test_otel_logging(self):
        config = {
            "algorithm": {"name": "ppo"},
            "data": {"train_files": "dummy_dataset"}
        }
        # Initialize Tracking with otel
        tracking = Tracking(
            project_name="test_project",
            experiment_name="test_experiment",
            default_backend=["otel"],
            config=config
        )

        # Log some data
        data = {
            "perf/throughput": 100.0,
            "perf/total_num_tokens": 5000,
            "timing_s/step": 5.0,
            "critic/score/mean": 0.8,
            "actor/loss": 0.1,
            "critic/loss": 0.2,
        }
        
        tracking.log(data, step=1)

        # For Gauges (ObservableGauge), we need to trigger collection
        # InMemoryMetricReader collects during get_metrics_data()
        metrics_data = self.reader.get_metrics_data()
        
        self.assertIsNotNone(metrics_data)
        
        # Parse metrics
        found_metrics = {}
        for resource_metrics in metrics_data.resource_metrics:
            for scope_metrics in resource_metrics.scope_metrics:
                for metric in scope_metrics.metrics:
                    found_metrics[metric.name] = metric

        # Verify mapped metrics exist
        self.assertIn("rl.perf.throughput", found_metrics)
        self.assertIn("rl.train.tokens", found_metrics)
        self.assertIn("rl.loop.duration", found_metrics)
        self.assertIn("rl.environment.reward.mean", found_metrics)
        self.assertIn("rl.train.loss", found_metrics)

        # Verify counter value
        tokens_metric = found_metrics["rl.train.tokens"]
        # In OTel Python SDK, Counter data point is usually accessible
        point = tokens_metric.data.data_points[0]
        self.assertEqual(point.value, 5000)

        # Verify gauge values
        throughput_metric = found_metrics["rl.perf.throughput"]
        point = throughput_metric.data.data_points[0]
        self.assertEqual(point.value, 100.0)

        print("✅ OTel Tracking Verification Passed!")

if __name__ == "__main__":
    unittest.main()
