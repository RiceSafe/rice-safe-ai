FROM python:3.10-slim

# Install system dependencies
USER root
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Setup the Hugging Face User (ID 1000)
RUN useradd -m -u 1000 user
WORKDIR /app

# Handle Python dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=user . .

# Switch to the non-root user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Expose port (Hugging Face default is 7860)
EXPOSE 7860

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]
