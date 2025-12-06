FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY deployment/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY deployment/api/ ./api/
COPY models/ ./models/
COPY src/ ./src/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]

