import logging
import time
import random
import sys
from json import dumps

# --- Configuration ---
LOG_FILE = 'inference.log'
MODEL_NAME = 'fraud_detector_v1'
# ---------------------

# Set up a logger
log = logging.getLogger('inference-logger')
log.setLevel(logging.INFO)

# Write to a file
file_handler = logging.FileHandler(LOG_FILE)
formatter = logging.Formatter('%(message)s') # Just log the message itself
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

# Also print to console
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
log.addHandler(stream_handler)


def get_mock_metrics():
    """Generates a single, realistic log entry."""
    confidence = random.uniform(0.75, 1.0)
    inference_time = random.uniform(15.0, 100.0)
    
    # Simulate a 5% error rate
    is_error = random.random() < 0.05
    
    log_entry = {
        "model_name": MODEL_NAME,
        "model_version": "1.2.0",
        "type": "model_inference",
        "prediction_confidence": round(confidence, 4),
        "prediction_class": "legitimate" if confidence > 0.85 else "suspicious",
        "inference_time_ms": round(inference_time, 2),
        "cpu_usage": round(random.uniform(20.0, 60.0), 2),
        "memory_usage_mb": random.randint(500, 1500),
        "environment": "production",
        "request_id": f"req_{random.randint(10000, 99999)}",
        "error": is_error,
        "error_message": "NullPointerException" if is_error else None
    }
    return log_entry

if __name__ == "__main__":
    print(f"Generating realistic inference logs...")
    print(f"Writing to {LOG_FILE}. Press Ctrl+C to stop.")
    try:
        while True:
            # Get a new log entry
            log_data = get_mock_metrics()
            
            # Log it as a JSON string
            log.info(dumps(log_data))
            
            # Wait for 1-3 seconds
            time.sleep(random.uniform(1.0, 3.0))
            
    except KeyboardInterrupt:
        print("\nLog generation stopped.")