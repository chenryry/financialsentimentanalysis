from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"


@dataclass(frozen=True)
class SentenceScore:
    text: str
    label: str
    positive: float
    negative: float
    neutral: float

    @property
    def signed(self) -> float:
        return self.positive - self.negative

    def to_dict(self) -> dict:
        return asdict(self)


def device_label() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _load_model():
    device = torch.device(device_label())
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    id2label = {int(i): lbl.lower() for i, lbl in model.config.id2label.items()}
    return tokenizer, model, device, id2label


def score_sentences(sentences: list[str], batch_size: int = 16) -> list[SentenceScore]:
    if not sentences:
        return []
    tokenizer, model, device, id2label = _load_model()
    out: list[SentenceScore] = []
    with torch.inference_mode():
        for start in range(0, len(sentences), batch_size):
            batch = sentences[start : start + batch_size]
            enc = tokenizer(
                batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
            ).to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            for text, row in zip(batch, probs):
                scores = {id2label[i]: float(v) for i, v in enumerate(row)}
                label = max(scores, key=scores.get)
                out.append(
                    SentenceScore(
                        text=text,
                        label=label,
                        positive=scores.get("positive", 0.0),
                        negative=scores.get("negative", 0.0),
                        neutral=scores.get("neutral", 0.0),
                    )
                )
    return out
