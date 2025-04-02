import logging
import os

log_dir = 'data/logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

def setup_logger():
    logger = logging.getLogger('SecurityAndSurveillance')  # Use a common logger name
    logger.setLevel(logging.DEBUG)  # Capture all log levels from DEBUG and above
    
    # Create log format
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    
    # Create handlers
    console_handler = logging.StreamHandler()  # Logs to console
    file_handler = logging.FileHandler(os.path.join(log_dir, 'system.log'))  # Logs to file
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logger()
