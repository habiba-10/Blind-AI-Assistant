# web_app/model_service.py
"""
Model service: load YOLOv8 once and expose run_detection_from_bytes.
Returns:
 - detections: list of {label, confidence, bbox: [x1,y1,x2,y2]}
 - vis_rel: relative path under static/ (e.g., "vis/xxx.jpg")
"""
import logging
from pathlib import Path
import time
import uuid
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO
import easyocr
from monitoring.metrics_logger import MetricsLogger

logger = logging.getLogger("model_service")
logger.setLevel(logging.INFO)

BASE_DIR = Path(__file__).resolve().parent
STATIC_VIS_DIR = BASE_DIR / "static" / "vis"
STATIC_VIS_DIR.mkdir(parents=True, exist_ok=True)

metrics_logger = MetricsLogger()

# 1. Load YOLO 
try:
    logger.info("Loading YOLOv8x (This might take time)...")
    yolo_model = YOLO("yolov8x.pt") 
    logger.info("YOLO model loaded.")
except Exception as e:
    yolo_model = None
    logger.error(f"Failed to load YOLO: {e}")

# 2. Load OCR 
try:
    logger.info("Loading OCR model...")
    reader = easyocr.Reader(['en'], gpu=False)
    logger.info("OCR model loaded.")
except Exception as e:
    reader = None
    logger.error(f"Failed to load OCR: {e}")

def run_detection(img_bytes: bytes):
    if yolo_model is None: return [], ""
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        results = yolo_model(img, verbose=False)
        r = results[0]
        
        detections = []
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = yolo_model.names[cls]
            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "bbox": box.xyxy[0].tolist()
            })
            # Log
            metrics_logger.log_prediction("object_detection", label, conf)

        # Visual
        vis_rel = ""
        try:
            res_plotted = r.plot()
            im_pil = Image.fromarray(res_plotted[..., ::-1])
            fname = f"det_{int(time.time())}_{uuid.uuid4().hex[:6]}.jpg"
            im_pil.save(STATIC_VIS_DIR / fname)
            vis_rel = f"vis/{fname}"
        except: pass

        return detections, vis_rel
    except Exception as e:
        logger.error(f"Detection Error: {e}")
        return [], ""

def run_ocr(img_bytes: bytes):
    if reader is None: return {"text": "OCR System Error"}, ""
    try:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # EasyOCR Inference
        result_list = reader.readtext(img, detail=0, paragraph=True)
        full_text = " ".join(result_list)
        
        if not full_text.strip(): full_text = "No text found"
        
        metrics_logger.log_prediction("ocr", "read_text", 1.0)
        return {"text": full_text}, ""
    except Exception as e:
        logger.error(f"OCR Error: {e}")
        return {"text": "Error reading text"}, ""

def process_image(img_bytes: bytes, mode: str):
    """ Router to switch between YOLO and OCR """
    if mode == "ocr":
        return run_ocr(img_bytes)
    else:
        return run_detection(img_bytes)