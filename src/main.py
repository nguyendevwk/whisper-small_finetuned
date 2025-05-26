import argparse
import os
import torch
import numpy as np
from datasets import DatasetDict
from dataset import load_and_prepare_dataset
from model import initialize_model_and_processor
from trainer import setup_trainer
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Whisper model on custom audio dataset")
    # Dataset parameters
    parser.add_argument('--train_metadata_csv', type=str, required=True,
                        help='Path to the metadata CSV file for training data')
    parser.add_argument('--eval_metadata_csv', type=str, required=True,
                        help='Path to the metadata CSV file for evaluation data')
    parser.add_argument('--audio_dir', type=str, required=True,
                        help='Path to the directory containing audio files')
    parser.add_argument('--dataset_name', type=str, default="custom_dataset",
                        help='Pretty name for the dataset (default: custom_dataset)')

    # Model parameters
    parser.add_argument('--model_name', type=str, default="openai/whisper-small",
                        help='Pre-trained Whisper model to use (default: openai/whisper-small)')
    parser.add_argument('--language', type=str, default="vi",
                        help='Language for tokenizer and model (default: vi)')
    parser.add_argument('--task', type=str, default="transcribe",
                        help='Task for tokenizer and model (default: transcribe)')

    # Training parameters
    parser.add_argument('--output_dir', type=str, default="./whisper-small-vi",
                        help='Output directory for model checkpoints (default: ./whisper-small-vi)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Per-device batch size for training and evaluation (default: 8)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help='Number of gradient accumulation steps (default: 2)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate for training (default: 2e-4)')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='Number of warmup steps (default: 100)')
    parser.add_argument('--num_train_epochs', type=int, default=5,
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--max_steps', type=int, default=-1,
                        help='Maximum number of training steps (default: -1, use epochs)')
    parser.add_argument('--logging_steps', type=int, default=10,
                        help='Frequency of logging steps (default: 10)')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Frequency of saving checkpoints (default: 500)')
    parser.add_argument('--eval_steps', type=int, default=None,
                        help='Frequency of evaluation steps (default: None, use eval_strategy)')
    parser.add_argument('--eval_strategy', type=str, default="epoch",
                        choices=["no", "steps", "epoch"],
                        help='Evaluation strategy (default: epoch)')
    parser.add_argument('--save_strategy', type=str, default="epoch",
                        choices=["no", "steps", "epoch"],
                        help='Save strategy (default: epoch)')
    parser.add_argument('--logging_strategy', type=str, default="steps",
                        choices=["no", "steps", "epoch"],
                        help='Logging strategy (default: steps)')
    parser.add_argument('--load_best_model_at_end', action='store_true', default=True,
                        help='Load the best model at the end of training (default: True)')
    parser.add_argument('--metric_for_best_model', type=str, default="wer",
                        help='Metric to select the best model (default: wer)')
    parser.add_argument('--greater_is_better', action='store_true', default=False,
                        help='Whether higher metric values are better (default: False for WER)')
    parser.add_argument('--fp16', action='store_true', default=torch.cuda.is_available(),
                        help='Use FP16 mixed precision training (default: auto based on GPU)')
    parser.add_argument('--gradient_checkpointing', action='store_true', default=True,
                        help='Use gradient checkpointing to save memory (default: True)')

    # Generation parameters
    parser.add_argument('--generation_max_length', type=int, default=225,
                        help='Maximum length for generated sequences (default: 225)')
    parser.add_argument('--num_beams', type=int, default=2,
                        help='Number of beams for beam search during generation (default: 2)')
    parser.add_argument('--predict_with_generate', action='store_true', default=True,
                        help='Use generate() for evaluation predictions (default: True)')

    # Callback parameters
    parser.add_argument('--enable_sample_predictions', action='store_true', default=True,
                        help='Enable sample prediction callback during evaluation (default: True)')
    parser.add_argument('--num_sample_predictions', type=int, default=3,
                        help='Number of sample predictions to show during evaluation (default: 3)')

    # Miscellaneous
    parser.add_argument('--push_to_hub', action='store_true', default=False,
                        help='Push model to Hugging Face Hub (default: False)')
    parser.add_argument('--report_to', type=str, nargs='+', default=["tensorboard"],
                        help='Reporting tools for logging (default: tensorboard)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    return parser.parse_args()

def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, using CPU")

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load and prepare dataset
    dataset = load_and_prepare_dataset(
        args.train_metadata_csv,
        args.eval_metadata_csv,
        args.audio_dir,
        args.model_name,
        args.language,
        args.task
    )

    # Initialize model and processor
    model, processor, tokenizer = initialize_model_and_processor(
        args.model_name,
        args.language,
        args.task
    )

    # Setup trainer
    trainer = setup_trainer(
        model,
        processor,
        tokenizer,
        dataset,
        args
    )

    # Save processor
    processor.save_pretrained(args.output_dir)

    print("🚀 Starting training...")
    if args.enable_sample_predictions:
        print(f"📋 Will show {args.num_sample_predictions} sample predictions after each evaluation")

    trainer.train()

    # Save the final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n🎉 Training completed!")
    print(f"💾 Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()