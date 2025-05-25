import logging
import os

def setup_logging():
    """Configure logging with file and console output."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('finetune_whisper.log'),
            logging.StreamHandler()
        ]
    )

def validate_file(path: str, file_type: str, is_dir: bool = False):
    """Validate existence of a file or directory."""
    logger = logging.getLogger(__name__)
    check_func = os.path.isdir if is_dir else os.path.isfile
    if not check_func(path):
        logger.error(f"{file_type} not found at: {path}")
        raise FileNotFoundError(f"{file_type} not found at: {path}")
    logger.info(f"{file_type} validated at: {path}")