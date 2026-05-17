FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (Docker cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port uvicorn will listen on
EXPOSE 8000

# Start the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
