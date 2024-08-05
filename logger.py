import os 
import sys 
import logging
from rich.logging import RichHandler 

logger_str = '[%(asctime)s: %(levelname)s: %(module)s: %(message)s]'

log_dir = 'logs'

log_filename = os.path.join(log_dir,'logs.log')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level='NOTSET',
    format=logger_str,
    handlers=[
        logging.FileHandler(log_filename),
        RichHandler()
    ]
    
)

log = logging.getLogger('yaml-pipe')
