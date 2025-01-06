import logging
import os
from datetime import datetime

# Generate the log filename
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the logs directory
logs_dir = os.path.join(os.getcwd(), "src/logs")
os.makedirs(logs_dir, exist_ok=True)  # Create only the directory

# Full log file path
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)

# if __name__ == "__main__":
#     logging.info("This is a log message")  # Test log message
