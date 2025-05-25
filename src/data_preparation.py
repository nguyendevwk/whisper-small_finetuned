from datasets import DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import logging
import torch
import numpy as np

def prepare_dataset(dataset: DatasetDict, feature_extractor: WhisperFeatureExtractor,
                  tokenizer: WhisperTokenizer) -> DatasetDict:
    """Prepare dataset for Whisper model training with optimized batched processing."""
    logger = logging.getLogger(__name__)

    def process_batch(batch):
        """Process a batch of examples for feature extraction and tokenization."""
        try:
            input_features = []
            labels = []

            for idx, (audio_dict, transcription) in enumerate(zip(batch["audio"], batch["transcription"])):
                try:
                    audio_array = audio_dict["array"]
                    sampling_rate = audio_dict["sampling_rate"]

                    if not isinstance(audio_array, (np.ndarray, list)) or sampling_rate != 16000:
                        logger.warning(f"Skipping invalid audio sample at index {idx}: invalid data or sampling rate {sampling_rate}")
                        continue

                    with torch.no_grad():
                        features = feature_extractor(
                            audio_array,
                            sampling_rate=sampling_rate,
                            return_tensors="pt"
                        ).input_features.squeeze(0)

                    label_ids = tokenizer(transcription).input_ids
                    if not label_ids:
                        logger.warning(f"Skipping sample at index {idx}: empty transcription")
                        continue

                    input_features.append(features)
                    labels.append(label_ids)

                except Exception as e:
                    logger.warning(f"Error processing sample at index {idx}: {str(e)}")
                    continue

            if not input_features or not labels:
                logger.warning("No valid samples in batch, returning empty batch")
                return {"input_features": [], "labels": []}

            return {"input_features": input_features, "labels": labels}

        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise

    logger.info("Preparing dataset with single-threaded processing...")
    processed_dataset = dataset.map(
        process_batch,
        remove_columns=dataset.column_names["train"],
        num_proc=1,  # Single-threaded for Kaggle
        desc="Preparing dataset",
        batch_size=100,  # Small batch size for memory efficiency
        batched=True
    )

    def is_valid_example(example):
        return len(example["input_features"]) > 0 and len(example["labels"]) > 0

    processed_dataset = processed_dataset.filter(is_valid_example, num_proc=1, desc="Filtering invalid examples")

    return processed_dataset