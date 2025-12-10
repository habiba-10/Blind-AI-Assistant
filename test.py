from monitoring.metrics_logger import MetricsLogger

m = MetricsLogger()
result = m.log_prediction("object_detection", "chair", 0.87, True)
print(result)
