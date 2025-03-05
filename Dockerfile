FROM python:3.9

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

CMD ["uvicorn", "src.fastapi_service:app", "--host", "0.0.0.0", "--port", "8000"]
