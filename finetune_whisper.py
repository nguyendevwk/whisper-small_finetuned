import argparse
import os
import torch
import random
import numpy as np
import logging
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer
)
from src.data_preparation import prepare_dataset
from src.trainer_config import get_training_args, DataCollatorSpeechSeq2SeqWithPadding
from src.evaluation import compute_metrics, SamplePredictionCallback
from src.utils import setup_logging

def parse_arguments():
    """Parse command-line arguments for fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model on custom audio dataset")
    parser.add_argument('--metadata_csv', type=str, required=True,
                        help='Path to metadata CSV file containing audio paths and transcriptions')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Path to directory containing audio files')
    parser.add_argument('--output_dir', type=str, default="./whisper-small-vi",
                        help='Output directory for model checkpoints (default: ./whisper-small-vi)')
    parser.add_argument('--dataset_name', type=str, default="custom_dataset",
                        help='Pretty name for the dataset (default: custom_dataset)')
    parser.add_argument('--num_sample_predictions', type=int, default=3,
                        help='Number of sample predictions to show during evaluation (default: 3)')
    parser.add_argument('--show_samples_every', type=int, default=100,
                        help='Show sample predictions every N training steps (default: 100)')
    parser.add_argument('--model_name', type=str, default="openai/whisper-small",
                        help='Hugging Face pretrained model name (default: openai/whisper-small)')
    return parser.parse_args()

def main():
    """Main function for fine-tuning Whisper model."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Log file structure for debugging
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Files in current directory: {os.listdir('.')}")

    # Parse arguments
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Log hardware info
    logger.info(f"GPU available: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU, using CPU'}")

    # Set random seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)

    # Load and prepare dataset
    logger.info("Loading dataset...")
    dataset = load_dataset('csv', data_files={'train': args.metadata_csv}, delimiter=',')['train']

    def update_audio_path(example):
        """Update audio paths to absolute paths."""
        example['audio'] = os.path.join(args.audio_dir, example['audio'])
        return example

    dataset = dataset.map(update_audio_path, desc="Updating audio paths")

    # Split dataset
    infore1 = DatasetDict()
    splits = dataset.train_test_split(train_size=0.8, test_size=0.1, seed=101, shuffle=True)
    splits_test = splits['test'].train_test_split(train_size=0.5, test_size=0.5, seed=101, shuffle=True)
    infore1["train"] = splits["train"]
    infore1["validation"] = splits_test["train"]
    infore1["test"] = splits_test["test"]
    logger.info(f"Dataset splits: {infore1}")

    # Initialize Whisper components
    logger.info(f"Loading pretrained model: {args.model_name}")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_name)
    tokenizer = WhisperTokenizer.from_pretrained(args.model_name, language="vi", task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.model_name, language="vi", task="transcribe")

    # Prepare dataset
    infore1 = infore1.cast_column("audio", Audio(sampling_rate=16000))
    infore1 = prepare_dataset(infore1, feature_extractor, tokenizer)

    # Initialize model
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    model.generation_config.language = "vi"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    # Setup training arguments
    training_args = get_training_args(args.output_dir)

    # Initialize data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Initialize sample prediction callback
    sample_callback = SamplePredictionCallback(
        eval_dataset=infore1["validation"],
        processor=processor,
        tokenizer=tokenizer,
        num_samples=args.num_sample_predictions,
        show_every_n_steps=args.show_samples_every
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=infore1["train"],
        eval_dataset=infore1["validation"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer),
        processing_class=processor,  # Fix FutureWarning
        callbacks=[sample_callback]
    )

    # Save processor
    processor.save_pretrained(args.output_dir)

    logger.info("Starting training...")
    logger.info(f"Will show {args.num_sample_predictions} sample predictions:")
    logger.info(f"  • Every {args.show_samples_every} training steps")
    logger.info(f"  • After each evaluation")

    # Train model
    trainer.train()

    # Save final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Training completed!")
    logger.info(f"Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()