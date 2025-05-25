import argparse
import os
import torch
import logging
from datasets import load_dataset, DatasetDict, Audio
from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from src.data_preparation import prepare_dataset
from src.trainer_config import get_training_args, DataCollatorSpeechSeq2SeqWithPadding
from src.evaluation import compute_metrics, EvaluationCallback
from src.utils import setup_logging, validate_file

def parse_arguments():
    """Parse and validate command-line arguments for fine-tuning."""
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model on custom audio dataset")
    parser.add_argument('--metadata_csv', type=str, required=True,
                        help='Path to metadata CSV file containing audio paths and transcriptions')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Path to directory containing audio files')
    parser.add_argument('--output_dir', type=str, default="./whisper-small-vi",
                        help='Output directory for model checkpoints (default: ./whisper-small-vi)')
    parser.add_argument('--dataset_name', type=str, default="custom_dataset",
                        help='Pretty name for the dataset (default: custom_dataset)')
    parser.add_argument('--selection_mode', type=str, choices=['epoch', 'step'], default='epoch',
                        help='Select checkpoint by epoch or step (default: epoch)')
    parser.add_argument('--checkpoint', type=int,
                        help='Specific checkpoint to evaluate (epoch or step number)')
    parser.add_argument('--model_name', type=str, default="openai/whisper-small",
                        help='Hugging Face pretrained model name (default: openai/whisper-small)')
    return parser.parse_args()

def main():
    """Main function for fine-tuning Whisper model."""
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    # Parse arguments
    args = parse_arguments()

    # Validate input files
    validate_file(args.metadata_csv, "Metadata CSV")
    validate_file(args.audio_dir, "Audio directory", is_dir=True)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Log hardware info
    logger.info(f"GPU available: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU, using CPU'}")

    # Load and prepare dataset
    logger.info("Loading dataset...")
    dataset = load_dataset('csv', data_files={'train': args.metadata_csv}, delimiter=',')['train']

    def update_audio_path(example):
        """Update audio paths to absolute paths."""
        audio_path = os.path.join(args.audio_dir, example['audio'])
        validate_file(audio_path, f"Audio file {audio_path}")
        example['audio'] = audio_path
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
    training_args = get_training_args(args.output_dir, args.selection_mode)

    # Initialize data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=infore1["train"],
        eval_dataset=infore1["validation"],
        data_collator=data_collator,
        compute_metrics=lambda pred: compute_metrics(pred, tokenizer),
        tokenizer=processor.feature_extractor,
        callbacks=[EvaluationCallback(infore1["test"], processor, tokenizer, args.output_dir)],
    )

    # Save processor
    processor.save_pretrained(args.output_dir)

    # Train or evaluate specific checkpoint
    if args.checkpoint is None:
        logger.info("Starting training...")
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    else:
        checkpoint_path = f"{args.output_dir}/checkpoint-{args.checkpoint}"
        if args.selection_mode == 'epoch':
            checkpoint_path = f"{args.output_dir}/checkpoint-epoch-{args.checkpoint}"
        logger.info(f"Evaluating checkpoint: {checkpoint_path}")
        from evaluation import evaluate_and_visualize
        evaluate_and_visualize(trainer, infore1["test"], processor, tokenizer, checkpoint_path)

if __name__ == "__main__":
    main()