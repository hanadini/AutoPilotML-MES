FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY artifacts ./artifacts

EXPOSE 19830

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "19830", "--app-dir", "src"]
