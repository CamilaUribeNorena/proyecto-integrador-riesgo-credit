FROM python:3.13-slim

WORKDIR /app

# Dependencias primero: aprovecha la cache de capas de Docker
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt


# Código y artefactos necesarios para servir
COPY src/ src/
COPY models/ models/
COPY data/raw/ data/raw/

EXPOSE 8000

CMD ["uvicorn", "src.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]