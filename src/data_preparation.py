from datasets import DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import logging
import torch

def prepare_dataset(dataset: DatasetDict, feature_extractor: WhisperFeatureExtractor,
                  tokenizer: WhisperTokenizer) -> DatasetDict:
    """Prepare dataset for Whisper model training with optimized processing."""
    logger = logging.getLogger(__name__)

    def process_batch(batch):
        """Process a single batch for feature extraction and tokenization."""
        try:
            audio = batch["audio"]
            # Use torch.no_grad() to save memory during feature extraction
            with torch.no_grad():
                batch["input_features"] = feature_extractor(
                    audio["array"],
                    sampling_rate=audio["sampling_rate"],
                    return_tensors="pt"
                ).input_features.squeeze(0)
            batch["labels"] = tokenizer(batch["transcription"]).input_ids
            return batch
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise

    logger.info("Preparing dataset with multiprocessing...")
    return dataset.map(
        process_batch,
        remove_columns=dataset.column_names["train"],
        num_proc=max(2, os.cpu_count() // 2),  # Optimize number of processes
        desc="Preparing dataset",
        batch_size=1000,  # Process in batches to reduce memory overhead
        batched=True
    )