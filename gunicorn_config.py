"""
gunicorn_config.py

Purpose
- Production-ready baseline configuration for Gunicorn when serving the Flask API.

How it fits
- Used by container/runtime process managers to start the web app; tuned conservatively for
	responsiveness with moderate concurrency.

Main role
- Define worker/thread counts, bind address/port, timeouts, and logging for the Gunicorn server.
"""
# These settings are a solid starting point for a production environment.

# Number of worker processes. A good rule of thumb is (2 x $num_cores) + 1
workers = 5

# The number of threads for each worker
threads = 2

# The socket to bind to. A non-privileged port is used by default.
bind = '0.0.0.0:8000'

# Timeout for workers in seconds
timeout = 120

# The log level
loglevel = 'info'