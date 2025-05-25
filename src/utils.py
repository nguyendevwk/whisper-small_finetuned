import logging

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