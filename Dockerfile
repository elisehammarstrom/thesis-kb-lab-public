FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including fonts for proper text rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libfontconfig1 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directory structure
RUN mkdir -p /app/data_exploration /data /output/data_exploration /output/data_exploration/visualizations /images_jpg

# Copy the OCR analysis script to its own directory
COPY data_exploration/data_exploration.py /app/data_exploration/

# Set the entrypoint
ENTRYPOINT ["python", "/app/data_exploration/data_exploration.py"]

# Default command line arguments with specific output path
CMD ["--base-path", "/data","--image-path", "/images_jpg", "--output", "/output/data_exploration/results.json", "--visualize", "--viz-dir", "/output/data_exploration/visualizations"]