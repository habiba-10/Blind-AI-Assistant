# web_app/main.py
"""
FastAPI app for AI Vision for the Blind.
Run from project root:
    uvicorn web_app.main:app --reload

Serves:
 - GET /        -> index UI
 - POST /predict-> accepts multipart/form-data file (image) and returns JSON detections + visualization path
 - GET /health  -> basic health
"""
import logging
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from web_app.model_service import process_image, yolo_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("web_app")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "vis").mkdir(parents=True, exist_ok=True)

app = FastAPI(title="AI Vision for the Blind - API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(mode: str = "object", file: UploadFile = File(...)):
    """
    Accepts mode='object' or 'ocr'
    """
    try:
        contents = await file.read()
        result, vis_rel = process_image(contents, mode)
        
        return JSONResponse(content={
            "result": result, 
            "visualization": f"/static/{vis_rel}" if vis_rel else ""
        })
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": yolo_model is not None}