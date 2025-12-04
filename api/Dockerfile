FROM python:3.10-slim

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU-only PyTorch from the CPU wheel index, then other packages from PyPI
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir "numpy<2" fastapi uvicorn transformers

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
