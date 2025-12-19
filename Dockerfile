##############################################
# Stage 1 — Builder
##############################################
FROM python:3.10-slim AS builder

# Install build tools (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirement file
COPY requirements.txt .

# Install dependencies into a wheel folder
RUN pip install --upgrade pip \
    && pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

##############################################
# Stage 2 — Final Light Runtime Image
##############################################
FROM python:3.10-slim AS runtime

# Install only runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy wheels from builder
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install wheels (offline install)
RUN pip install --no-cache /wheels/*

# Copy your model + code
COPY tinybert_dual_classifier_quantized.onnx .
COPY vocab.json .
COPY tokenizer_config.json .
COPY labels.json .
COPY testing.py .

# Default command
CMD ["python", "testing.py"]
