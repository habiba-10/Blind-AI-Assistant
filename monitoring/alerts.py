#alerts.py
import smtplib
import requests
import yaml
import os
from email.mime.text import MIMEText

BASE = os.path.dirname(__file__)
with open(os.path.join(BASE, "config.yaml"), "r") as f:
    CONFIG = yaml.safe_load(f)

def send_email_alert(subject, message):
    if not CONFIG["notifications"]["email"]["enabled"]:
        return
    creds = CONFIG["notifications"]["email"]
    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = creds["sender_email"]
    msg["To"] = ", ".join(creds["recipients"])
    with smtplib.SMTP(creds["smtp_server"], creds["smtp_port"]) as server:
        server.starttls()
        server.login(creds["sender_email"], creds["sender_password"])
        server.send_message(msg)

def send_slack_alert(message):
    if not CONFIG["notifications"]["slack"]["enabled"]:
        return
    webhook = CONFIG["notifications"]["slack"]["webhook_url"]
    requests.post(webhook, json={"text": message})

def trigger_alert(metric_summary):
    acc = metric_summary.get("avg_accuracy", 1)
    conf = metric_summary.get("avg_confidence", 1)
    thresholds = CONFIG["alert_threshold"]
    if acc is not None and acc < thresholds["accuracy_drop"]:
        send_email_alert("Accuracy Drop Alert", f"Model accuracy dropped to {acc:.2f}")
        send_slack_alert(f"⚠ Model accuracy dropped to {acc:.2f}")
    if conf < thresholds["confidence_drop"]:
        send_email_alert("Confidence Drop Alert", f"Average confidence dropped to {conf:.2f}")
        send_slack_alert(f"⚠ Average confidence dropped to {conf:.2f}")