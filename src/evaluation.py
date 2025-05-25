import evaluate
import torch
import logging
from transformers import Seq2SeqTrainer, WhisperProcessor, WhisperTokenizer
from datasets import Dataset
import os

def compute_metrics(pred, tokenizer):
    """Compute Word Error Rate (WER) for model predictions."""
    metric = evaluate.load("wer")
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

class EvaluationCallback:
    """Callback for evaluating and visualizing test set predictions."""
    def __init__(self, test_dataset: Dataset, processor: WhisperProcessor,
                 tokenizer: WhisperTokenizer, output_dir: str):
        self.test_dataset = test_dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

    def on_evaluate(self, args, state, control, **kwargs):
        """Evaluate and visualize predictions after each epoch/step."""
        checkpoint_dir = f"{self.output_dir}/checkpoint-{state.global_step}"
        if args.eval_strategy == "epoch":
            checkpoint_dir = f"{self.output_dir}/checkpoint-epoch-{state.epoch}"

        self.logger.info(f"Evaluating checkpoint at {checkpoint_dir}")
        trainer = kwargs['trainer']
        with torch.no_grad():  # Save memory during evaluation
            predictions = trainer.predict(self.test_dataset)

        # Log sample predictions (first 5 samples)
        pred_ids = predictions.predictions[:5]
        label_ids = predictions.label_ids[:5]
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        self.logger.info("\nSample Predictions:")
        for i, (pred, ref) in enumerate(zip(pred_str, label_str)):
            self.logger.info(f"Sample {i+1}:")
            self.logger.info(f"Predicted: {pred}")
            self.logger.info(f"Reference: {ref}")
            self.logger.info("-" * 50)

        wer = compute_metrics(predictions, self.tokenizer)
        self.logger.info(f"WER for checkpoint {checkpoint_dir}: {wer['wer']:.2f}%")

def evaluate_and_visualize(trainer: Seq2SeqTrainer, test_dataset: Dataset,
                         processor: WhisperProcessor, tokenizer: WhisperTokenizer,
                         checkpoint_path: str):
    """Evaluate a specific checkpoint and visualize results."""
    logger = logging.getLogger(__name__)

    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint path {checkpoint_path} does not exist")
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} does not exist")

    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    trainer.model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
    with torch.no_grad():  # Save memory during evaluation
        predictions = trainer.predict(test_dataset)

    # Visualize predictions
    pred_ids = predictions.predictions[:5]
    label_ids = predictions.label_ids[:5]
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    logger.info(f"\nPredictions for checkpoint {checkpoint_path}:")
    for i, (pred, ref) in enumerate(zip(pred_str, label_str)):
        logger.info(f"Sample {i+1}:")
        logger.info(f"Predicted: {pred}")
        logger.info(f"Reference: {ref}")
        logger.info("-" * 50)

    wer = compute_metrics(predictions, tokenizer)
    logger.info(f"WER for checkpoint {checkpoint_path}: {wer['wer']:.2f}%")