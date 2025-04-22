import logging

# Logger config
logger = logging.getLogger("scene-monitor")
logger.setLevel(logging.DEBUG)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# File handler
fh = logging.FileHandler("monitor.log")
fh.setLevel(logging.INFO)

# Formatter
formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# Attach handlers
logger.addHandler(ch)
logger.addHandler(fh)