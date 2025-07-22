FROM python:3.11-slim

# Install OpenCV & Tesseract with multiple language packs
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-mar \
    tesseract-ocr-ben \
    tesseract-ocr-kan \
    tesseract-ocr-tam \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy app files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Streamlit
EXPOSE 5000

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=5000", "--server.address=0.0.0.0"]
