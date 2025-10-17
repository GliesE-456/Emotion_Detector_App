# backend/model.py
import logging
from typing import List, Dict, Any
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import threading
import time

LOG = logging.getLogger(__name__)

_model = None
_lock = threading.Lock()

# Choose a compact emotion model (smaller than big roberta)
# We picked: j-hartmann/emotion-english-distilroberta-base
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"


def get_pipeline():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                try:
                    LOG.info("Loading transformer model %s ...", MODEL_NAME)
                    # return_all_scores=False gives top predictions; set top_k for multiple outputs
                    _model = pipeline("text-classification", model=MODEL_NAME, top_k=None)
                    LOG.info("Model loaded.")
                except Exception as e:
                    LOG.exception("Failed to load transformer model: %s", e)
                    _model = None
    return _model


# Simple lexicon fallback (if transformer unavailable)
EMO_LEX = {
    "joy": ["happy", "glad", "joy", "delight", "love", "smile", "excited"],
    "sadness": ["sad", "unhappy", "depressed", "sorrow", "down", "lonely", "mourn"],
    "anger": ["angry", "furious", "mad", "hate", "annoyed", "irritated", "rage"],
    "fear": ["afraid", "scared", "fear", "fright", "terrified", "panic", "anxious"],
    "surprise": ["surprised", "shocked", "wow", "astonished", "unexpected"],
    "neutral": []
}


def predict_texts(texts: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Predict emotions for a list of texts.
    Returns list of dicts: {"predictions":[{"label": label, "score":score}, ...]}
    """
    pipe = get_pipeline()
    results = []
    if pipe:
        try:
            # when top_k=None the pipeline returns scores for all labels
            raw = pipe(texts)
            # pipeline shapes: single input -> list of dicts OR for batch returns list per item
            # normalize
            if isinstance(raw, dict) or (len(raw) > 0 and isinstance(raw[0], dict)):
                raw = [raw]
            for r in raw:
                # r maybe list of dicts (label, score) or single dict
                if isinstance(r, list):
                    # sort by score desc and take top_k
                    sorted_preds = sorted(r, key=lambda x: -x.get("score", 0.0))
                    if top_k:
                        sorted_preds = sorted_preds[:top_k]
                    results.append({"predictions": sorted_preds})
                else:
                    results.append({"predictions": [r]})
            return results
        except Exception as e:
            LOG.exception("Transformer prediction failed: %s", e)

    # Fallback lexicon
    import re
    from collections import Counter
    for t in texts:
        tokens = re.findall(r"\w+", t.lower())
        counts = Counter()
        for emo, words in EMO_LEX.items():
            for w in words:
                counts[emo] += sum(1 for tok in tokens if tok == w)
        if sum(counts.values()) == 0:
            results.append({"predictions": [{"label": "neutral", "score": 1.0}]})
        else:
            total = sum(counts.values())
            sorted_em = sorted(counts.items(), key=lambda kv: -kv[1])
            preds = [{"label": k, "score": v/total} for k, v in sorted_em]
            results.append({"predictions": preds[:top_k]})
    return results