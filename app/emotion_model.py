from transformers import pipeline
import re

# Try to load the transformer model
try:
    emotion_analyzer = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)
    MODEL_AVAILABLE = True
except Exception:
    emotion_analyzer = None
    MODEL_AVAILABLE = False

# Simple lexicon fallback
EMO_LEX = {
    "joy": ["happy", "glad", "joy", "delight", "love", "smile", "excited", "pleased", "content", "cheerful"],
    "sadness": ["sad", "unhappy", "depressed", "sorrow", "down", "lonely", "mourn", "gloomy", "melancholy", "blue"],
    "anger": ["angry", "furious", "mad", "hate", "annoyed", "irritated", "rage", "outraged", "resentful", "agitated"],
    "fear": ["afraid", "scared", "fear", "fright", "terrified", "panic", "anxious", "nervous", "worried", "apprehensive"],
    "surprise": ["surprised", "shocked", "wow", "astonished", "unexpected", "amazed", "startled", "stunned"],
    "disgust": ["disgust", "gross", "nasty", "repulsed", "revolted", "sickened", "abhorrent", "offended", "distaste"],
    "trust": ["trust", "confident", "secure", "faith", "reliable", "dependable", "assured"],
    "anticipation": ["anticipate", "expect", "hopeful", "eager", "excited", "await", "predict"],
    "neutral": []
}

def lexicon_analyze(text: str):
    text_lower = text.lower()
    scores = {k: 0 for k in EMO_LEX}
    for emotion, keywords in EMO_LEX.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", text_lower):
                scores[emotion] += 1
    top_emotion = max(scores, key=scores.get)
    if scores[top_emotion] == 0:
        top_emotion = "neutral"
    return {
        "label": top_emotion,
        "score": 1.0 if top_emotion != "neutral" else 0.0,
        "all": [{"label": k, "score": v} for k, v in scores.items()]
    }

def analyze_text(text: str):
    """
    Analyze the emotion of a given text using a pre-trained transformer model.
    Falls back to a lexicon-based method if the model is unavailable.
    Returns a dict with label, score, and all predictions.
    """
    if not text.strip():
        return {"error": "Empty text"}
    if MODEL_AVAILABLE:
        try:
            results = emotion_analyzer(text)[0]
            results.sort(key=lambda x: x["score"], reverse=True)
            top = results[0]
            return {"label": top["label"], "score": round(top["score"], 3), "all": results}
        except Exception as e:
            # Fallback to lexicon if model fails
            return lexicon_analyze(text) | {"warning": f"Model error: {e}"}
    else:
        return lexicon_analyze(text) | {"warning": "Transformer model unavailable, using lexicon fallback."}