"""
1단계/2단계 분류 모델을 로드하고 예측 함수를 제공하는 헬퍼.

추후 파일 저장소(Dify 레포) 이관 시:
    core/helper/agent_api_helper/classifier_api_helper.py 로 경로만 옮기면 됩니다.
    (기존 membership_api_helper.py 와 동일한 패턴 — Tool 코드는 이 클래스의
    인터페이스만 알고, joblib/scikit-learn을 직접 알지 못합니다.)
"""

import os
import joblib

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)

STAGE1_MODEL_PATH = os.path.join(ROOT_DIR, "models", "stage1_classifier.joblib")
STAGE2_MODEL_PATH = os.path.join(ROOT_DIR, "models", "stage2_classifier.joblib")


class ClassifierHelper:
    def __init__(self):
        self._stage1 = None
        self._stage2 = None

    def _ensure_loaded_stage1(self):
        if self._stage1 is None:
            if not os.path.exists(STAGE1_MODEL_PATH):
                raise FileNotFoundError(
                    f"{STAGE1_MODEL_PATH} 없음. scripts/train_stage1.py 를 먼저 실행하세요."
                )
            self._stage1 = joblib.load(STAGE1_MODEL_PATH)

    def _ensure_loaded_stage2(self):
        if self._stage2 is None:
            if not os.path.exists(STAGE2_MODEL_PATH):
                raise FileNotFoundError(
                    f"{STAGE2_MODEL_PATH} 없음. scripts/train_stage2.py 를 먼저 실행하세요."
                )
            self._stage2 = joblib.load(STAGE2_MODEL_PATH)

    def classify_stage1(self, text: str) -> dict:
        self._ensure_loaded_stage1()
        pred = self._stage1.predict([text])[0]
        proba = max(self._stage1.predict_proba([text])[0])
        return {"category": pred, "confidence": round(float(proba), 4)}

    def classify_stage2(self, text: str) -> dict:
        self._ensure_loaded_stage2()
        pred = self._stage2.predict([text])[0]
        proba = max(self._stage2.predict_proba([text])[0])
        return {"category": pred, "confidence": round(float(proba), 4)}


classifierHelper = ClassifierHelper()
