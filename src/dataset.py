from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperFeatureExtractor, WhisperProcessor
import os

def load_and_prepare_dataset(train_metadata_csv, eval_metadata_csv, audio_dir, model_name, language, task):
    # Load datasets from CSV
    dataset = DatasetDict()
    dataset['train'] = load_dataset(
        'csv',
        data_files={'train': train_metadata_csv},
        delimiter=','
    )['train']
    dataset['validation'] = load_dataset(
        'csv',
        data_files={'validation': eval_metadata_csv},
        delimiter=','
    )['validation']

    # Update audio paths to be absolute
    def update_audio_path(example):
        audio_path = os.path.join(audio_dir, example['audio'])
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        example['audio'] = audio_path
        return example

    dataset = dataset.map(update_audio_path)

    print("Dataset splits:", dataset)

    # Initialize feature extractor and processor
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)

    # Cast audio column to 16kHz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Prepare dataset for training
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["transcription"]).input_ids
        return batch

    dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=2)

    return dataset