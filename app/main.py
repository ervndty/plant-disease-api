from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from utils.pipeline import inference_pipeline
from utils.logger import logger
from models.labels import LABELS

app = FastAPI(
    title="🌿 Plant Disease Prediction API",
    description="API untuk deteksi penyakit tanaman berbasis CNN feature extraction + ML pipeline",
    version="1.0.0"
)

@app.get("/")
def root():
    return {
        "status": "success",
        "message": "Plant Disease Prediction API is running."
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        logger.info(f"Received file: {file.filename}")
        pred, confidence = inference_pipeline(image_bytes)

        # pastikan tipe datanya normal
        if hasattr(pred, "item"):
            pred = pred.item()
        if hasattr(confidence, "item"):
            confidence = confidence.item()

        # 🧩 ubah index ke nama label
        disease_name = LABELS[pred] if isinstance(pred, int) and pred < len(LABELS) else str(pred)

        return JSONResponse({
            "status": "success",
            "result": {
                "disease": disease_name,
                "confidence": round(float(confidence) * 100, 2) if confidence else None
            }
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": str(e)
            }
        )
