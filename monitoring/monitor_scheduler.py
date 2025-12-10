# monitor_scheduler
import time
import yaml
import threading
from .metrics_logger import MetricsLogger
from .alerts import trigger_alert
from .retrain_manager import check_retraining_condition

with open("monitoring/config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

def run_monitoring_loop():
    logger = MetricsLogger()
    while True:
        metrics = logger._read_all()
        if metrics:
            summary = logger.log_prediction("summary", "check", 1.0)
            trigger_alert(summary)
            check_retraining_condition(summary)
        time.sleep(CONFIG["retrain"]["check_interval_minutes"] * 60)

def start_background_monitoring():
    thread = threading.Thread(target=run_monitoring_loop, daemon=True)
    thread.start()