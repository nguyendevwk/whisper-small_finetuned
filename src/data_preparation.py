from datasets import DatasetDict
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import logging
import torch
import numpy as np
import os

def prepare_dataset(dataset: DatasetDict, feature_extractor: WhisperFeatureExtractor,
                    tokenizer: WhisperTokenizer) -> DatasetDict:
    """Prepare dataset for Whisper model training with optimized batched processing (no filtering)."""
    logger = logging.getLogger(__name__)

    def process_batch(batch):
        """Process a batch of examples for feature extraction and tokenization."""
        input_features = []
        labels = []

        for audio_dict, transcription in zip(batch["audio"], batch["transcription"]):
            # Extract audio array and sampling rate
            audio_array = audio_dict.get("array", None)
            sampling_rate = audio_dict.get("sampling_rate", None)

            # Skip if data is missing or invalid (but do not remove sample from dataset)
            if not isinstance(audio_array, (np.ndarray, list)) or sampling_rate != 16000:
                logger.warning(f"Invalid or missing audio data: {audio_dict}")
                input_features.append(None)
                labels.append(None)
                continue

            try:
                with torch.no_grad():
                    features = feature_extractor(
                        audio_array,
                        sampling_rate=sampling_rate,
                        return_tensors="pt"
                    ).input_features.squeeze(0)

                input_features.append(features)
                labels.append(tokenizer(transcription).input_ids)
            except Exception as e:
                logger.warning(f"Error processing sample: {str(e)}")
                input_features.append(None)
                labels.append(None)

        return {
            "input_features": input_features,
            "labels": labels
        }

    logger.info("Preparing dataset without filtering (retaining all samples)...")
    processed_dataset = dataset.map(
        process_batch,
        remove_columns=dataset.column_names["train"],
        num_proc=max(2, os.cpu_count() // 2),
        desc="Preparing dataset",
        batch_size=100,
        batched=True
    )

    return processed_dataset
