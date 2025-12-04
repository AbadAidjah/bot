FROM python:3.10-slim

WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU-only PyTorch + other packages
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir "numpy<2" transformers datasets accelerate peft

COPY train.py .
COPY cs_data.json .

# Use unbuffered Python output so logs show in real-time
ENV PYTHONUNBUFFERED=1

CMD ["python3", "train.py"]
