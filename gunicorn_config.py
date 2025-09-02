# gunicorn_config.py
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