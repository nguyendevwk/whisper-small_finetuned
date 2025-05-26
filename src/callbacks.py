from transformers import TrainerCallback
import torch
import random

class SamplePredictionCallback(TrainerCallback):
    def __init__(self, eval_dataset, processor, tokenizer, num_samples=3):
        self.eval_dataset = eval_dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.num_samples = num_samples

        # Chọn random samples một lần duy nhất
        self.sample_indices = random.sample(range(len(eval_dataset)), min(num_samples, len(eval_dataset)))
        self.sample_data = [eval_dataset[i] for i in self.sample_indices]

    def on_evaluate(self, args, state, control, model, **kwargs):
        """Được gọi sau mỗi lần evaluation"""
        print("\n" + "="*80)
        print(f"📊 SAMPLE PREDICTIONS - Step {state.global_step} | Epoch {state.epoch:.1f}")
        print("="*80)

        model.eval()
        with torch.no_grad():
            for i, sample in enumerate(self.sample_data):
                try:
                    # Prepare input
                    input_features = torch.tensor(sample["input_features"]).unsqueeze(0)
                    if torch.cuda.is_available():
                        input_features = input_features.cuda()
                        model = model.cuda()

                    # Generate prediction
                    predicted_ids = model.generate(
                        input_features,
                        max_length=225,
                        num_beams=2,
                        early_stopping=True
                    )

                    # Decode prediction
                    prediction = self.tokenizer.decode(predicted_ids[0], skip_special_tokens=True)

                    # Get ground truth
                    if "labels" in sample:
                        # Convert labels back to text
                        labels = sample["labels"].copy()
                        labels = [l if l != -100 else self.tokenizer.pad_token_id for l in labels]
                        ground_truth = self.tokenizer.decode(labels, skip_special_tokens=True)
                    else:
                        ground_truth = "N/A"

                    # Display results
                    print(f"\n🎯 Sample {i+1} (Index: {self.sample_indices[i]}):")
                    print(f"📝 Ground Truth: {ground_truth}")
                    print(f"🤖 Prediction:   {prediction}")

                    # Calculate simple character-level similarity
                    if ground_truth != "N/A":
                        similarity = self._calculate_similarity(ground_truth.lower(), prediction.lower())
                        print(f"📈 Similarity:   {similarity:.1f}%")

                except Exception as e:
                    print(f"❌ Error processing sample {i+1}: {str(e)}")

        print("\n" + "="*80 + "\n")

    def _calculate_similarity(self, text1, text2):
        """Tính độ tương đồng đơn giản giữa 2 chuỗi"""
        if not text1 or not text2:
            return 0.0

        # Simple character-level similarity
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 100.0

        # Count matching characters at same positions
        matches = sum(1 for a, b in zip(text1, text2) if a == b)
        return (matches / max_len) * 100