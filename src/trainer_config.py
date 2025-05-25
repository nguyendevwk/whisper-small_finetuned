from dataclasses import dataclass
from typing import Any, Dict, List, Union
import torch
from transformers import Seq2SeqTrainingArguments

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data collator for speech sequence-to-sequence models with padding."""
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """Collate and pad features for batch processing."""
        # Pad input features
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove decoder start token if present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def get_training_args(output_dir: str, selection_mode: str) -> Seq2SeqTrainingArguments:
    """Configure optimized training arguments for Whisper model."""
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        # Batch size optimized for GPU memory
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,  # Effective batch size of 16
        # Learning rate optimized for fine-tuning
        learning_rate=1e-5,  # Conservative LR for pretrained model
        warmup_steps=100,    # Reduced warmup for faster convergence
        # Flexible training duration
        num_train_epochs=5 if selection_mode == 'epoch' else -1,
        max_steps=4000 if selection_mode == 'step' else -1,
        # Memory optimization
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        # Evaluation and saving strategy
        eval_strategy=selection_mode,
        save_strategy=selection_mode,
        logging_strategy="steps",
        logging_steps=10,
        save_steps=500 if selection_mode == 'step' else -1,
        # Model selection
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        # Generation settings
        predict_with_generate=True,
        generation_max_length=225,
        # Logging
        report_to=["tensorboard"],
        push_to_hub=False,
        # Additional optimizations
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,  # Use BF16 on compatible GPUs
        optim="adamw_torch",  # Stable optimizer
        dataloader_num_workers=2,  # Optimize data loading
        remove_unused_columns=False,  # Keep columns for custom processing
    )