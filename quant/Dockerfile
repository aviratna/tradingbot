FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY quant/requirements_quant.txt ./requirements_quant.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements_quant.txt

# Copy the full project (needed for app/ imports)
COPY . .

# Create data directories
RUN mkdir -p quant/data/snapshots

# Run the quant engine
CMD ["python", "-m", "quant.main"]
