from celery import Celery
import os

# Configure Redis URL. Use environment variable or default.
# Ensure your Redis server is running at this address/port.
REDIS_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")

# Create the Celery application instance
app = Celery(
    'inference_tasks', # Name of the Celery application
    broker=REDIS_URL,
    backend=REDIS_URL, # Use Redis as the result backend as well
    include=['tasks'] # List of modules to import when the worker starts
)

# Optional Celery configuration settings
app.conf.update(
    task_serializer='json',      # Use json for task serialization
    result_serializer='json',    # Use json for result serialization
    accept_content=['json'],     # Accept json content
    timezone='UTC',              # Set timezone
    enable_utc=True,             # Enable UTC
    # Configure task tracking and result expiration if needed
    # result_expires=3600,       # Example: expire results after 1 hour
    # task_track_started=True,   # To get 'STARTED' state (requires result backend)
)

if __name__ == '__main__':
    # This allows running the worker directly using: python celery_app.py worker ...
    # Although usually you run: celery -A celery_app worker ...
    app.start() 