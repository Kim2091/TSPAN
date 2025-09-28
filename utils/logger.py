import logging
import sys

def setup_logger(name='TSPAN', level=logging.INFO):
    """Set up and return a logger with a custom format."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        # Logger is already configured, don't add handlers again
        return logger

    logger.setLevel(level)
    
    # Create a handler that writes to stdout
    handler = logging.StreamHandler(sys.stdout)
    
    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%y-%m-%d %H:%M')
    handler.setFormatter(formatter)
    
    # Add the handler to the logger
    logger.addHandler(handler)
    
    return logger

# Create a default logger instance to be imported by other modules
logger = setup_logger()
