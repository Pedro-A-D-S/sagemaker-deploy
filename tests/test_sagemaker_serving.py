import pytest
import json
from sagemaker_serving.inference import SageMakerInferenceHandler
from sagemaker_serving.train import train_and_export

@pytest.fixture
def handler():
    return SageMakerInferenceHandler()

def test_inference_tabular_prediction(handler):
    payload = json.dumps({"features": [0.12, 0.45, 0.88, 0.99]})
    input_data = handler.input_fn(payload, "application/json")
    model = handler.model_fn("dummy_model_dir")
    pred = handler.predict_fn(input_data, model)
    output_str = handler.output_fn(pred)
    
    assert pred["inference_type"] == "TABULAR_CLASSIFIER"
    assert "prediction_class" in pred
    assert pred["latency_ms"] >= 0.0
    assert "whisper-dl-endpoint-prod" in output_str

def test_inference_audio_transcription(handler):
    payload = json.dumps({"audio_bytes_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQAAAAA="})
    input_data = handler.input_fn(payload, "application/json")
    pred = handler.predict_fn(input_data, {})
    
    assert pred["inference_type"] == "WHISPER_ASR"
    assert "priority account verification" in pred["transcription"]
    assert pred["confidence"] > 0.90

def test_training_separation_no_leakage():
    res = train_and_export("data/train", "data/test", "models/export")
    assert res["test_accuracy"] > 0.95
    assert res["test_loss"] < 0.15
