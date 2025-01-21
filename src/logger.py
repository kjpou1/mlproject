import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler

# Create a timestamped log file
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the directory to store log files
logs_dir = os.path.join(os.getcwd(), "logs")

# Ensure the directory exists
os.makedirs(logs_dir, exist_ok=True)

# Full path to the log file
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Set up dynamic log level (default to INFO if not specified)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()

# Configure logging
formatter = logging.Formatter(
    "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s"
)

# Create a rotating file handler
handler = RotatingFileHandler(
    LOG_FILE_PATH, maxBytes=5 * 1024 * 1024, backupCount=5  # 5 MB per file, 5 backups
)
handler.setFormatter(formatter)

# Get the root logger and configure it
logger = logging.getLogger()
logger.setLevel(log_level)  # Set dynamic log level
logger.addHandler(handler)

# Optionally, log to console as well
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Test log messages (optional for debugging)
if __name__ == "__main__":
    logger.info("Hic est nuntius INFO.")
    logger.debug("HHic est nuntius DEBUG .")
    logger.error("Hic est nuntius ERROR.")
