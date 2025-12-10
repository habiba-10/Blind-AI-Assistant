# mlflow_tracking.py
import os
import argparse
import platform
import socket
from datetime import datetime, timezone
import logging
from pathlib import Path
import traceback
import json

import mlflow
import mlflow.pytorch

import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mlflow_tracking")

# Constants and folders
DEFAULT_EXPERIMENT = "Blind_Vision_Object_Detection"
BASE_DIR = Path.cwd()
MLRUNS_DIR = BASE_DIR / "mlruns"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
MODELS_DIR = BASE_DIR / "models"

for d in (MLRUNS_DIR, ARTIFACTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def collect_system_info():
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "hostname": socket.gethostname(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["cuda_available"] = "unknown"
    return info


def draw_and_save_result(result, out_path: Path):
    try:
        img_plot = result.plot()
        if isinstance(img_plot, np.ndarray):
            img = Image.fromarray(img_plot)
        else:
            img = img_plot
        img.save(out_path)
        return True
    except Exception as e:
        logger.warning(f"Could not save visualization to {out_path}: {e}")
        return False


def run_quick_demo(model: YOLO, sample_images: list, run_dir: Path):
    metrics = {}
    total_detections = 0
    total_conf = 0.0
    total_images = 0
    image_stats = []

    for img_path in sample_images:
        total_images += 1
        try:
            logger.info(f"Inferencing image: {img_path}")
            results = model(img_path)
            r = results[0]
            boxes = getattr(r, "boxes", None)
            confs = []
            if boxes is not None and hasattr(boxes, "conf"):
                confs = boxes.conf.cpu().numpy().tolist()
            num = len(confs)
            total_detections += num
            total_conf += sum(confs) if confs else 0.0
            avg_conf = (sum(confs) / num) if num > 0 else 0.0
            image_stats.append({"image": str(img_path), "detections": num, "avg_conf": float(avg_conf)})

            vis_name = run_dir / f"vis_{Path(img_path).stem}.jpg"
            saved = draw_and_save_result(r, vis_name)
            if not saved:
                logger.warning(f"Visualization not saved for {img_path}")
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Inference failed for {img_path}: {e}\n{tb}")
            image_stats.append({"image": str(img_path), "error": str(e), "traceback": tb})

    metrics["total_images"] = total_images
    metrics["total_detections"] = total_detections
    metrics["mean_detections_per_image"] = (total_detections / total_images) if total_images > 0 else 0.0
    metrics["mean_confidence"] = (total_conf / total_detections) if total_detections > 0 else 0.0
    metrics["image_stats"] = image_stats
    return metrics


def try_log_torch_model_to_mlflow(model, model_name="yolov8_model"):
    try:
        torch_model = getattr(model, "model", None)
        if torch_model is not None:
            mlflow.pytorch.log_model(torch_model, artifact_path=model_name)
            logger.info("Logged torch model via mlflow.pytorch.log_model")
            return True
    except Exception as e:
        logger.warning(f"mlflow.pytorch.log_model failed: {e}")

    # fallback: save state_dict
    try:
        weights_path = MODELS_DIR / f"{model_name}_state_dict.pt"
        sd = getattr(model, "model", None)
        if sd is not None:
            torch.save(sd.state_dict(), weights_path)
            mlflow.log_artifact(str(weights_path), artifact_path="model_weights")
            logger.info(f"Saved and logged state_dict to {weights_path}")
            return True
    except Exception as e:
        logger.warning(f"Saving state_dict failed: {e}")

    return False


def main(args):
    # setup mlflow
    mlflow.set_tracking_uri(f"file:{MLRUNS_DIR.resolve()}")
    mlflow.set_experiment(args.experiment_name)

    run = None
    try:
        # Start run
        run = mlflow.start_run(run_name=args.run_name)
        run_id = run.info.run_id
        logger.info(f"MLflow run started: id={run_id}")
        print(">> MLflow run started")

        # system info
        system_info = collect_system_info()
        mlflow.log_dict(system_info, "system_info.json")
        mlflow.log_param("script", Path(__file__).name)
        mlflow.log_param("model_source", args.model_source)
        mlflow.log_param("evaluate_on_val", str(args.evaluate))

        # load model
        logger.info(f"Loading YOLO model: {args.model_source}")
        print(">> Loading YOLO model...")
        model = YOLO(args.model_source)
        print(">> Model loaded")

        # create run artifacts dir
        run_artifact_dir = ARTIFACTS_DIR / f"run_{run_id}"
        run_artifact_dir.mkdir(parents=True, exist_ok=True)

        # quick demo if images provided
        if args.sample_images:
            logger.info("Running quick demo on provided sample images")
            print(">> Running quick demo on sample images")
            demo_metrics = run_quick_demo(model, args.sample_images, run_artifact_dir)
            mlflow.log_metrics({
                "demo_total_images": demo_metrics["total_images"],
                "demo_total_detections": demo_metrics["total_detections"],
                "demo_mean_detections_per_image": demo_metrics["mean_detections_per_image"],
                "demo_mean_confidence": demo_metrics["mean_confidence"],
            })
            mlflow.log_dict(demo_metrics["image_stats"], "image_stats.json")
            # log visualizations
            for f in run_artifact_dir.iterdir():
                if f.suffix.lower() in {".jpg", ".png"}:
                    mlflow.log_artifact(str(f), artifact_path="visualizations")
            print(">> Quick demo complete and artifacts logged")

        if args.evaluate:
            try:
                print(">> Running model.val() (may take long)...")
                results = model.val(data=args.data) if args.data else model.val()
                try:
                    mlflow.log_metric("mAP50", float(getattr(results.box, "map50", float("nan"))))
                    mlflow.log_metric("mAP50_95", float(getattr(results.box, "map", float("nan"))))
                except Exception:
                    logger.warning("Could not extract mAP metrics from results object.")
                mlflow.log_text(str(results), "val_results.txt")
                print(">> model.val() finished")
            except Exception as e:
                tb = traceback.format_exc()
                logger.error(f"Full evaluation failed: {e}\n{tb}")
                mlflow.log_param("evaluation_error", str(e))
                mlflow.log_artifact(run_artifact_dir / "evaluation_failed.txt")

        logged = try_log_torch_model_to_mlflow(model, model_name="yolov8n")
        mlflow.log_param("model_logged", str(logged).lower())

        report = {
            "run_id": run.info.run_id,
            "model_source": args.model_source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        mlflow.log_dict(report, "run_report.json")

        print(">> MLflow run completed successfully")
        logger.info("MLflow run completed successfully")
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Unhandled exception in main: {e}\n{tb}")
        print(">> ERROR: Unhandled exception - details logged")
        try:
            mlflow.log_param("run_error", str(e))
            err_path = ARTIFACTS_DIR / f"error_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt"
            err_path.write_text(tb)
            if mlflow.active_run():
                mlflow.log_artifact(str(err_path), artifact_path="errors")
        except Exception as inner_e:
            logger.warning(f"Failed to log error to MLflow: {inner_e}")
    finally:
        try:
            if mlflow.active_run():
                mlflow.end_run()
                logger.info("MLflow run ended (end_run called).")
        except Exception:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLflow tracking for YOLOv8 demo")
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--run_name", type=str, default="yolov8_pretrained_demo")
    parser.add_argument("--model_source", type=str, default="yolov8n.pt")
    parser.add_argument("--sample_images", nargs="*", default=[
        r"C:\Users\ezath\OneDrive\Desktop\Blind_project\chair.jpg",
        r"C:\Users\ezath\OneDrive\Desktop\Blind_project\door.jpg"
    ])
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--data", type=str, default="")
    args = parser.parse_args()

    # resolve sample image paths
    sample_imgs = []
    for s in args.sample_images:
        p = Path(s)
        if p.exists():
            sample_imgs.append(str(p.resolve()))
        else:
            logger.warning(f"Sample image not found, skipping: {s}")
    args.sample_images = sample_imgs

    main(args)