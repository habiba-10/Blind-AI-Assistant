#retrain_manager
import os
import subprocess
import yaml

BASE = os.path.dirname(__file__)
with open(os.path.join(BASE, "config.yaml"), "r") as f:
    CONFIG = yaml.safe_load(f)

def check_retraining_condition(summary):
    retrain_cfg = CONFIG["retrain"]
    records = summary.get("count", 0)
    acc = summary.get("avg_accuracy", 1)
    if acc is not None and acc < CONFIG["alert_threshold"]["accuracy_drop"]:
        trigger_retraining(records)

def trigger_retraining(records):
    retrain_cfg = CONFIG["retrain"]
    if records < retrain_cfg["min_records_for_retrain"]:
        return
    cmd = ["python", retrain_cfg["retrain_script"]]
    subprocess.Popen(cmd)