"""
modules/text_detector.py
Module 1: Text Fake News Detection
Model : hamzab/roberta-fake-news-classification (HuggingFace)
"""

from transformers import pipeline
import torch

MODEL_NAME       = "hamzab/roberta-fake-news-classification"
MAX_TOKEN_LENGTH = 512
_classifier      = None   # module-level cache


def load_model():
    """Load RoBERTa pipeline once, then cache it in memory."""
    global _classifier
    if _classifier is None:
        print(f"[text_detector] Loading '{MODEL_NAME}' ...")
        device = 0 if torch.cuda.is_available() else -1
        _classifier = pipeline(
            task="text-classification",
            model=MODEL_NAME,
            tokenizer=MODEL_NAME,
            device=device,
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
        )
        print(f"[text_detector] Ready on {'GPU' if device == 0 else 'CPU'}.")
    return _classifier


def predict(text: str) -> dict:
    """
    Classify one news text.
    Returns: { "label": "FAKE"|"REAL", "confidence": float, "raw": list }
    """
    if not text or not text.strip():
        raise ValueError("Input text cannot be empty.")
    clf        = load_model()
    raw_output = clf(text.strip())          # [{'label': 'FAKE', 'score': 0.97}]
    top        = raw_output[0]
    return {
        "label":      top["label"].upper(),
        "confidence": round(top["score"], 4),
        "raw":        raw_output,
    }


def predict_batch(texts: list) -> list:
    """Classify a list of texts in one batched call."""
    if not texts:
        return []
    clf         = load_model()
    raw_outputs = clf([t.strip() for t in texts], batch_size=8)
    return [
        {"label": r["label"].upper(), "confidence": round(r["score"], 4), "raw": [r]}
        for r in raw_outputs
    ]


def confidence_tier(confidence: float) -> str:
    """Turn a score into a readable label."""
    if confidence >= 0.90: return "Very High"
    if confidence >= 0.75: return "High"
    if confidence >= 0.55: return "Moderate"
    return "Low"


if __name__ == "__main__":
    samples = [
        "NASA confirms astronauts will return to the Moon by 2026.",
        "Government secretly puts 5G chips inside COVID vaccines.",
        "The Federal Reserve raised interest rates by 0.25 percent.",
        "Doctors HATE this weird trick that CURES all diseases instantly!",
    ]
    for text in samples:
        r = predict(text)
        icon = "🔴" if r["label"] == "FAKE" else "🟢"
        print(f"{icon} {r['label']} | {confidence_tier(r['confidence'])} | {r['confidence']:.1%}")
        print(f"   {text[:75]}\n")