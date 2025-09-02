# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# --no-cache-dir makes the image smaller
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Expose the port that Gunicorn will run on
EXPOSE 8000

# Define the command to run your application
# It tells Gunicorn to run the 'app' object from your 'index.py' file, using our new config.
CMD ["gunicorn", "-c", "gunicorn_config.py", "api.index:app"]