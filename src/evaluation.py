import evaluate
import torch
import logging
import random
from transformers import TrainerCallback
from transformers import Seq2SeqTrainer, WhisperProcessor, WhisperTokenizer
from datasets import Dataset
from typing import Any, Dict, List, Union

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

class SamplePredictionCallback(TrainerCallback):
    """Callback to display sample predictions during training and evaluation."""
    def __init__(self, eval_dataset: Dataset, processor: WhisperProcessor,
                 tokenizer: WhisperTokenizer, num_samples: int = 3, show_every_n_steps: int = 100):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.show_every_n_steps = show_every_n_steps
        self.last_shown_step = -1
        self.logger = logging.getLogger(__name__)

        # Select random samples once
        self.sample_indices = random.sample(range(len(eval_dataset)), min(num_samples, len(eval_dataset)))
        self.sample_data = [eval_dataset[i] for i in self.sample_indices]

    def on_log(self, args, state, control, model, logs=None, **kwargs):
        """Show predictions on log events."""
        if logs is None or state.global_step <= 0 or state.global_step == self.last_shown_step:
            return control

        if 'loss' in logs and state.global_step % self.show_every_n_steps == 0:
            self.last_shown_step = state.global_step
            self._show_sample_predictions(model, state, "🔄 TRAINING")

        return control

    def on_evaluate(self, args, state, control, model, logs=None, **kwargs):
        """Show predictions after evaluation."""
        self._show_sample_predictions(model, state, "📊 EVALUATION")
        return control

    def _show_sample_predictions(self, model, state, phase_name):
        """Display sample predictions."""
        self.logger.info(f"\n{'='*80}")
        self.logger.info(f"{phase_name} - Step {state.global_step} | Epoch {state.epoch:.2f}")
        self.logger.info('='*80)

        original_training = model.training
        model.eval()

        with torch.no_grad():
            for i, sample in enumerate(self.sample_data):
                try:
                    input_features = torch.tensor(sample["input_features"]).unsqueeze(0)
                    if torch.cuda.is_available() and next(model.parameters()).is_cuda:
                        input_features = input_features.cuda()

                    predicted_ids = model.generate(
                        input_features,
                        max_length=225,
                        num_beams=1,
                        do_sample=False,
                        early_stopping=True
                    )

                    prediction = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

                    labels = sample["labels"].copy()
                    labels = [l if l != -100 else self.tokenizer.pad_token_id for l in labels]
                    ground_truth = self.tokenizer.decode(labels, skip_special_tokens=True)

                    self.logger.info(f"\n🎯 Sample {i+1}:")
                    self.logger.info(f"📝 Truth: {ground_truth[:100]}{'...' if len(ground_truth) > 100 else ''}")
                    self.logger.info(f"🤖 Pred:  {prediction[:100]}{'...' if len(prediction) > 100 else ''}")

                    similarity = self._calculate_similarity(ground_truth.lower(), prediction.lower())
                    self.logger.info(f"📈 Match: {similarity:.1f}%")

                except Exception as e:
                    self.logger.warning(f"Error processing sample {i+1}: {str(e)[:50]}...")

        if original_training:
            model.train()

        self.logger.info('='*80 + '\n')

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple character-level similarity between two strings."""
        if not text1 or not text2:
            return 0.0
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 100.0
        matches = sum(1 for a, b in zip(text1, text2) if a == b)
        return (matches / max_len) * 100