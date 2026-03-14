# 1. Use the official, completely pristine PyTorch image directly from the source
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# 2. Install our required system graphics libraries
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Install our clean Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the RunPod Serverless handler
COPY handler.py .

# 5. Keep the unbuffered diagnostic flag to stream logs instantly
CMD ["python", "-u", "handler.py"]
