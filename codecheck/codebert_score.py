from __future__ import annotations

import numpy as np

_MODEL_NAME = "microsoft/codebert-base"


class CodeBERTScorer:
    """Embedding-consistency scorer (SelfCheck-CodeBERT). evaluate() returns the mean
    embedding dissimilarity of the samples vs the main implementation, on the same [0,1]
    scale and direction as Exec/Prompt/AST (higher = more likely incorrect).

    Each implementation is embedded with CodeBERT (mean-pooled, L2-normalized token
    states); per-sample dissimilarity = 1 - cosine(main, sample), clamped to [0,1]. The
    model is loaded lazily on first use (microsoft/codebert-base, ~500MB from HF). Unlike
    the AST metric this is NOT rename/format-invariant — it is a learned semantic/lexical
    similarity, which is the point of the comparison.
    """

    def __init__(self, model_name: str = _MODEL_NAME, device: str | None = None,
                 batch_size: int = 32) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._device = device
        self._tokenizer = None
        self._model = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device).eval()

    def _embed(self, codes: list[str]) -> np.ndarray:
        """L2-normalized mean-pooled CodeBERT embeddings, shape (len(codes), dim)."""
        import torch
        self._ensure_model()
        vectors = []
        for start in range(0, len(codes), self.batch_size):
            batch = codes[start:start + self.batch_size]
            enc = self._tokenizer(batch, return_tensors="pt", truncation=True,
                                  max_length=512, padding=True)
            enc = {k: v.to(self._device) for k, v in enc.items()}
            with torch.no_grad():
                states = self._model(**enc).last_hidden_state  # (B, T, H)
            mask = enc["attention_mask"].unsqueeze(-1)         # (B, T, 1)
            pooled = (states * mask).sum(1) / mask.sum(1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            vectors.append(pooled.cpu().numpy())
        return np.concatenate(vectors, axis=0)

    def evaluate(self, main_code: str, sample_codes: list[str]) -> tuple[float, list[float]]:
        """(mean_dissimilarity, per_sample_dissimilarities)."""
        if not sample_codes:
            return 0.0, []
        if not main_code or not main_code.strip():
            # The main is a non-answer; nothing to compare against -> maximally divergent.
            return 1.0, [1.0] * len(sample_codes)
        embeddings = np.asarray(self._embed([main_code] + list(sample_codes)), dtype=float)
        main_vec = embeddings[0]
        per_sample: list[float] = []
        for i, code in enumerate(sample_codes, start=1):
            if not code or not code.strip():
                per_sample.append(1.0)
                continue
            cosine = float(np.dot(main_vec, embeddings[i]))
            per_sample.append(min(1.0, max(0.0, 1.0 - cosine)))
        return sum(per_sample) / len(per_sample), per_sample

    def score(self, main_code: str, sample_codes: list[str]) -> float:
        return self.evaluate(main_code, sample_codes)[0]
