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

# ── 규칙 기반 오버라이드 ──────────────────────────────────────────────
# 우선 사용내역은 쓸지 말지 테스트를 계속 해보면서 진행을 함. 추후에 사용에 문제가 있다면 제외 시켜야함 제외는 ### 까지

# 아래 패턴에 해당하면 통계 모델(TF-IDF)을 거치지 않고 무조건 membership으로
# 확정한다. "사용내역 조회" / "과거 사용여부 확인" / "이용가능여부 확인"은
# 예외 없이 membership이라는 게 확정된 규칙이라, 데이터로 모델에 학습시키는
# 것보다 규칙으로 먼저 걸러내는 편이 안정적이다.

### 추후 제외 여기서부터 시작
_ACTION_STEMS = ["사용", "이용", "썼", "쓴", "써봤"]
_RECORD_WORDS = ["내역", "기록"]
_AVAILABILITY_PATTERNS = [
    r"사용\s*가능", r"이용\s*가능", r"쓸\s*수\s*있",
]
_STANDALONE_CONFIRM = ["썼", "써봤", "사용했", "이용했"]  # "썼어?"처럼 내역/기록 없이도 그 자체로 확정

_AVAILABILITY_REGEX = re.compile("|".join(_AVAILABILITY_PATTERNS))


def apply_membership_rules(text: str) -> bool:
    """사용내역/이용가능여부 확정 규칙에 해당하면 True (membership 강제)."""
    has_record = any(w in text for w in _RECORD_WORDS)
    has_action = any(w in text for w in _ACTION_STEMS)
    if has_record and has_action:
        return True
    if _AVAILABILITY_REGEX.search(text):
        return True
    if any(w in text for w in _STANDALONE_CONFIRM):
        return True
    if re.search(r"쓴\s", text):  # "쓴 내역" 외에 "내가 쓴 거" 같은 단독 표현
        return True
    return False

# === 아래는 우선적으로 돌려보는 곳 사용 내역과 같이 멤버십으로 무조건 분류되는 것들은 아래를 사용하지만,
### 추후 사용되는 규칙 중 문제 생길 시 위 펑션 및 규칙 값들 제거 필요!!!!!

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
        probs = self.predict_proba(text)
        best_label = max(probs, key=probs.get)
        return {"category": best_label, "confidence": round(probs[best_label], 4)}
