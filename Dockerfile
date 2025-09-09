# Dockerfile (Corrected and Optimized)

# SOLUTION 1: Upgrade to Python 3.12. This is the recommended fix.
# It solves the pandas-ta issue directly and keeps your project modern.
# The "-slim" variant is smaller and better for production.
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Good practice: Upgrade pip to the latest version
RUN pip install --no-cache-dir --upgrade pip

# Copy ONLY the requirements file first.
# This leverages Docker's layer caching. The packages will only be re-installed
# if you change the requirements.txt file itself.
COPY requirements.txt .

# Install all dependencies from the single requirements.txt file in one go.
# This is cleaner and more efficient.
RUN pip install --no-cache-dir -r requirements.txt

# Now, copy the rest of your application code.
# Any changes to your code from now on will use the cached layer above,
# making builds much faster.
COPY . .

# Expose the port
EXPOSE 8000

# Define the run command
CMD ["gunicorn", "-c", "gunicorn_config.py", "api.index:app"]