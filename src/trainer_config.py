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
        valid_features = [f for f in features if f["input_features"] is not None and f["labels"] is not None]
        if not valid_features:
            raise ValueError("No valid features in batch")

        input_features = [{"input_features": f["input_features"]} for f in valid_features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in valid_features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def get_training_args(output_dir: str) -> Seq2SeqTrainingArguments:
    """Configure optimized training arguments for Whisper model."""
    return Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        warmup_steps=100,
        num_train_epochs=5,
        max_steps=-1,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_strategy="steps",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to=["tensorboard"],
        push_to_hub=False,
        dataloader_num_workers=1,  # Reduced for Kaggle stability
        remove_unused_columns=False,
    )