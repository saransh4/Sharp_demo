# Use official Python base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy files into container
COPY . .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose port 8080
EXPOSE 8080

# Run the app
CMD ["python", "main.py"]
