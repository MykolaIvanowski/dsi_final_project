from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os
import io

app = FastAPI(title="Campaign Response API", version="1.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def find_file(filename):
    # Try ALL possible locations — local + Railway layouts
    candidates = [
        os.path.join(BASE_DIR, "model", filename),          # /app/model/ — Railway with COPY app/app.py ./app.py
        os.path.join(BASE_DIR, filename),                   # /app/final_model.pkl
        os.path.join(BASE_DIR, "..", "model", filename),    # dsi_final_project/model/ — local
        os.path.join("/app", "model", filename),             # /app/model/ — absolute
        os.path.join("/app", "app", "model", filename),     # /app/app/model/
        os.path.join(os.path.dirname(BASE_DIR), "model", filename),
    ]
    # Normalize .. in path
    candidates = [os.path.normpath(p) for p in candidates]
    for path in candidates:
        if os.path.exists(path):
            print(f"Found {filename} at: {path}")
            return path
    # If not found, print what we checked — for Railway logs
    print(f"NOT FOUND {filename}. Checked:")
    for p in candidates:
        print(f" - {p} exists={os.path.exists(p)}")
    # List what actually exists in /app for debugging
    try:
        for root, dirs, files in os.walk("/app"):
            print(f"DIR {root}: {files}")
    except Exception:
        pass
    return candidates[0]  # return first for error message

model = None
scaler = None
feature_columns = None
load_error = None
try:
    model = joblib.load(find_file("final_model.pkl"))
    scaler = joblib.load(find_file("scaler.pkl"))
    feature_columns = joblib.load(find_file("feature_columns.pkl"))
    print(f"Model loaded from: {find_file('final_model.pkl')}")
    print(f"Features: {feature_columns}")
except Exception as e:
    load_error = str(e)
    print(f"LOAD ERROR: {load_error}")

class CustomerInput(BaseModel):
    total_sales: float = Field(..., ge=0)
    num_unique_products: int = Field(..., ge=0)
    num_invoices: int = Field(..., ge=0)
    nps: float = Field(..., ge=0, le=10)
    n_comp: int = Field(..., ge=0)
    n_communications: int = Field(..., ge=0)
    loyalty: int = Field(..., ge=0, le=1)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names: strip spaces, lower case
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    X = df[feature_columns].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)
    if "total_sales" in X.columns:
        X["total_sales"] = np.log1p(X["total_sales"].clip(lower=0))
    return X

def predict_df(df: pd.DataFrame):
    X = prepare_features(df)
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return proba, pred

@app.get("/", response_class=HTMLResponse)
async def home():
    html_path = os.path.join(BASE_DIR, "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h3>API is running. Open /docs for Swagger UI.</h3>")

@app.get("/health")
async def health():
    if load_error:
        return {"status": "error", "detail": load_error}
    return {"status": "ok", "model": "XGBoost max_depth=4 n_estimators=200", "test_auc": 0.7203, "features": feature_columns}

@app.post("/predict")
async def predict_single(customer: CustomerInput):
    if load_error:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")
    try:
        df = pd.DataFrame([customer.dict()])
        proba, pred = predict_df(df)
        return {"probability": float(proba[0]), "predicted_class": int(pred[0]), "predicted_label": "Responder" if pred[0] == 1 else "Non-Responder", "features_used": feature_columns}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(file: UploadFile = File(...)):
    if load_error:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")
    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        if df.empty:
            raise ValueError("CSV file is empty")
        data_dir = os.path.join(BASE_DIR, "data")
        os.makedirs(data_dir, exist_ok=True)
        with open(os.path.join(data_dir, file.filename), "wb") as out:
            out.write(content)
        proba, pred = predict_df(df)
        df["probability"] = np.round(proba, 4)
        df["predicted_class"] = pred
        df["predicted_label"] = np.where(pred == 1, "Responder", "Non-Responder")
        summary = {"total_records": len(df), "predicted_responders": int((pred == 1).sum()), "predicted_non_responders": int((pred == 0).sum()), "mean_probability": float(np.mean(proba)), "responder_rate_predicted": float(np.mean(pred))}
        return {"summary": summary, "results": df.to_dict(orient="records")}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV or inputs: {str(e)}")




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)