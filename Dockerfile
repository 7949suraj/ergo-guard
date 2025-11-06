# ---- Base image ----
FROM python:3.10-slim

# ---- Set working directory ----
WORKDIR /app

# ---- Copy project files ----
COPY . /app

# ---- Install system dependencies (needed for OpenCV) ----
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python dependencies ----
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ---- Expose Streamlit port ----
EXPOSE 8501

# ---- Run Streamlit ----
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
