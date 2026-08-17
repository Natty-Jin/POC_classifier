"""
scikit-learn, pandas, joblib 없이 numpy만으로 TF-IDF + 로지스틱회귀를
추론하는 모듈. scripts/export_to_numpy.py 로 export된 JSON을 읽어서
동작합니다.

char_wb n-gram 토큰화는 scikit-learn의 CountVectorizer._char_wb_ngrams
구현을 그대로 재현합니다 (동일한 vocabulary/idf를 쓰려면 토큰화 로직이
정확히 같아야 합니다).

의존성: numpy, json (표준 라이브러리) 뿐.
"""

import json
import re
import numpy as np

_WHITESPACE_RE = re.compile(r"\s\s+")

# ── 규칙 기반 오버라이드 (현재 비활성화) ─────────────────────────────
# "사용내역/이용가능여부"면 무조건 membership이라는 규칙이었는데,
# "가족이랑 같이 썼는데 나도 쓸 수 있어?"처럼 여러 사람이 얽힌 사용 규칙
# 설명이 필요한 경우까지 같이 걸려버리는 경계 문제가 있어 껐다.
# 다시 켜려면 아래 _RULES_ENABLED 만 True로 바꾸면 된다 (로직 자체는 그대로 둠).
_RULES_ENABLED = False

_ACTION_STEMS = ["사용", "이용", "썼", "쓴", "써봤"]
_RECORD_WORDS = ["내역", "기록"]
_AVAILABILITY_PATTERNS = [
    r"사용\s*가능", r"이용\s*가능", r"쓸\s*수\s*있",
]
_STANDALONE_CONFIRM = ["썼", "써봤", "사용했", "이용했"]  # "썼어?"처럼 내역/기록 없이도 그 자체로 확정

_AVAILABILITY_REGEX = re.compile("|".join(_AVAILABILITY_PATTERNS))


def apply_membership_rules(text: str) -> bool:
    """사용내역/이용가능여부 확정 규칙에 해당하면 True (membership 강제).
    _RULES_ENABLED = False 인 동안은 항상 False를 반환해 모델 판단에 맡긴다."""
    if not _RULES_ENABLED:
        return False

    has_record = any(w in text for w in _RECORD_WORDS)
    has_action = any(w in text for w in _ACTION_STEMS)
    if has_record and has_action:
        return True
    if _AVAILABILITY_REGEX.search(text):
        return True
    if any(w in text for w in _STANDALONE_CONFIRM):
        return True
    if re.search(r"쓴\s", text):
        return True
    return False


def char_wb_ngrams(text: str, ngram_range: tuple[int, int]) -> list[str]:
    """scikit-learn CountVectorizer(analyzer='char_wb')와 동일한 토큰화.

    주의: sklearn TfidfVectorizer는 기본값 lowercase=True 이므로,
    학습 때와 동일하게 소문자 변환을 먼저 적용해야 vocabulary가 맞습니다.
    """
    text = text.lower()
    text = _WHITESPACE_RE.sub(" ", text)
    min_n, max_n = ngram_range
    ngrams = []
    for word in text.split():
        w = " " + word + " "
        w_len = len(w)
        for n in range(min_n, min(max_n + 1, w_len + 1)):
            offset = 0
            ngrams.append(w[offset:offset + n])
            while offset + n < w_len:
                offset += 1
                ngrams.append(w[offset:offset + n])
            if offset == 0:
                break
    return ngrams


class NumpyTfidfLogisticClassifier:
    """
    export_to_numpy.py 가 만든 JSON 아티팩트 하나를 로드해서
    TF-IDF 벡터화 + 로지스틱회귀 추론을 numpy만으로 수행.
    """

    def __init__(self, json_path: str):
        with open(json_path, "r", encoding="utf-8") as f:
            artifact = json.load(f)

        self.ngram_range = tuple(artifact["ngram_range"])
        self.vocabulary: dict[str, int] = artifact["vocabulary"]
        self.idf = np.array(artifact["idf"], dtype=np.float64)
        self.classes: list[str] = artifact["classes"]
        self.coef = np.array(artifact["coef"], dtype=np.float64)        # (1, V) 이진분류 기준
        self.intercept = np.array(artifact["intercept"], dtype=np.float64)  # (1,)
        self.vocab_size = len(self.vocabulary)

    def _vectorize(self, text: str) -> np.ndarray:
        tokens = char_wb_ngrams(text, self.ngram_range)

        counts = np.zeros(self.vocab_size, dtype=np.float64)
        for tok in tokens:
            idx = self.vocabulary.get(tok)
            if idx is not None:
                counts[idx] += 1.0

        # sklearn TfidfVectorizer 기본값: tf(raw count) * idf, 이후 L2 정규화
        tfidf = counts * self.idf
        norm = np.linalg.norm(tfidf)
        if norm > 0:
            tfidf = tfidf / norm
        return tfidf

    def predict_proba(self, text: str) -> dict[str, float]:
        x = self._vectorize(text)
        z = float(np.dot(self.coef[0], x) + self.intercept[0])
        # 이진분류: coef_는 classes_[1](양성 클래스) 기준
        p_positive = 1.0 / (1.0 + np.exp(-z))
        p_negative = 1.0 - p_positive
        return {
            self.classes[0]: p_negative,
            self.classes[1]: p_positive,
        }

    def classify(self, text: str) -> dict:
        # 규칙 우선 적용 — 사용내역/이용가능여부 확정 패턴이면 모델을 아예 안 거침
        if apply_membership_rules(text):
            return {"category": "membership", "confidence": 1.0, "matched_rule": True}

        probs = self.predict_proba(text)
        best_label = max(probs, key=probs.get)
        return {"category": best_label, "confidence": round(probs[best_label], 4), "matched_rule": False}