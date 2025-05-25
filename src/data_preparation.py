from datasets import DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import logging
import torch
import numpy as np
import os
def prepare_dataset(dataset: DatasetDict, feature_extractor: WhisperFeatureExtractor,
                  tokenizer: WhisperTokenizer) -> DatasetDict:
    """Prepare dataset for Whisper model training with optimized batched processing."""
    logger = logging.getLogger(__name__)

    def process_batch(batch):
        """Process a batch of examples for feature extraction and tokenization."""
        try:
            # Initialize lists for batched outputs
            input_features = []
            labels = []

            # Handle batched audio inputs (batch["audio"] is a list of dictionaries)
            for audio_dict, transcription in zip(batch["audio"], batch["transcription"]):
                try:
                    # Extract audio array and sampling rate
                    audio_array = audio_dict["array"]
                    sampling_rate = audio_dict["sampling_rate"]

                    # Validate audio data
                    if not isinstance(audio_array, (np.ndarray, list)) or sampling_rate != 16000:
                        logger.warning(f"Invalid audio data or sampling rate: {sampling_rate}")
                        continue

                    # Feature extraction with torch.no_grad for memory efficiency
                    with torch.no_grad():
                        features = feature_extractor(
                            audio_array,
                            sampling_rate=sampling_rate,
                            return_tensors="pt"
                        ).input_features.squeeze(0)

                    input_features.append(features)
                    labels.append(tokenizer(transcription).input_ids)

                except Exception as e:
                    logger.warning(f"Error processing audio sample: {str(e)}")
                    continue

            # Return batched features
            return {
                "input_features": input_features,
                "labels": labels
            }
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise

    logger.info("Preparing dataset with multiprocessing...")
    processed_dataset = dataset.map(
        process_batch,
        remove_columns=dataset.column_names["train"],
        num_proc=max(2, os.cpu_count() // 2),  # Optimize number of processes
        desc="Preparing dataset",
        batch_size=100,  # Reduced batch size to prevent memory issues
        batched=True
    )

    # Filter out any empty batches (due to failed samples)
    processed_dataset = processed_dataset.filter(lambda x: x["input_features"] is not None and x["labels"] is not None)

    return processed_dataset