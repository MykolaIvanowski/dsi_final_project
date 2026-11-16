FROM python:3.12

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy app code and model
COPY app/app.py ./app.py
COPY app/index.html ./index.html
COPY model/ ./model/
COPY app/data/ ./data/

# Railway provides PORT env variable
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]

