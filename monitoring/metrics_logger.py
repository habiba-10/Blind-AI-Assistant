#metrics_logger
import json
import os
from datetime import datetime
import threading
import yaml

BASE_DIR = os.path.dirname(__file__)
with open(os.path.join(BASE_DIR, "config.yaml"), "r") as f:
    CONFIG = yaml.safe_load(f)

LOG_PATH = os.path.abspath(os.path.join(BASE_DIR, CONFIG["log_file"]))
_lock = threading.Lock()

class MetricsLogger:
    def _init_(self):
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        if not os.path.exists(LOG_PATH):
            with open(LOG_PATH, "w") as f:
                json.dump([], f)

    def _read_all(self):
        try:
            with open(LOG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _write_all(self, data):
        with open(LOG_PATH, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def log_prediction(self, model_name, label, confidence, correct=None, additional=None):
        rec = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model": model_name,
            "label": label,
            "confidence": float(confidence),
            "correct": None if correct is None else bool(correct),
            "additional": additional or {}
        }
        with _lock:
            data = self._read_all()
            data.append(rec)
            self._write_all(data)

            confidences = [d.get("confidence", 0.0) for d in data]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            labeled = [d for d in data if d.get("correct") is not None]
            avg_acc = (sum(1 for d in labeled if d["correct"]) / len(labeled)) if labeled else None

        return {
            "count": len(data),
            "avg_confidence": avg_conf,
            "avg_accuracy": avg_acc
        }