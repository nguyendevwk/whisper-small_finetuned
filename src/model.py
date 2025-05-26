from transformers import (
    WhisperTokenizer,
    WhisperProcessor,
    WhisperForConditionalGeneration,
    WhisperFeatureExtractor
)

def initialize_model_and_processor(model_name, language, task):
    # Initialize Whisper components
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language=language, task=task)
    processor = WhisperProcessor.from_pretrained(model_name, language=language, task=task)

    # Load pre-trained Whisper model
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.language = language
    model.generation_config.task = task
    model.generation_config.forced_decoder_ids = None

    return model, processor, tokenizer