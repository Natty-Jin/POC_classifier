"""
1단계 분류 Tool — 상용 배포용, numpy만 사용 (scikit-learn/pandas/joblib 불필요).

--- 실제 이관 시 ---
from core.helper.agent_api_helper.classifier_api_helper import classifierApiHelper
from core.tools.entities.tool_entities import ToolInvokeMessage
from core.tools.tool.builtin_tool import BuiltinTool
from core.workflow.nodes.tool.exc import BuiltinToolApiError
from libs.ai_agent_common import is_debug
--------------------
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.numpy_classifier import NumpyTfidfLogisticClassifier

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "stage1_classifier_numpy.json",
)

DEBUG = os.environ.get("POC_DEBUG", "false").lower() == "true"


def is_debug():
    return DEBUG


class BuiltinTool:
    """실제 Dify BuiltinTool의 최소 자리표시자(POC 전용)."""

    def create_text_message(self, text: str):
        return text


class ClassifyStage1Tool(BuiltinTool):
    """
    Tool for classifying user utterance into knowledge/membership (stage1).
    numpy 전용 백엔드 사용 — scikit-learn/pandas/joblib 런타임 의존성 없음.
    """
    is_external_api: bool = False
    _model: NumpyTfidfLogisticClassifier | None = None

    @classmethod
    def _get_model(cls) -> NumpyTfidfLogisticClassifier:
        if cls._model is None:
            cls._model = NumpyTfidfLogisticClassifier(MODEL_PATH)
        return cls._model

    def _invoke(self, user_id: str, tool_parameters: dict):
        user_utterance = tool_parameters.get("user_utterance", "").strip()

        if is_debug():
            print("ClassifyStage1 user_utterance", user_utterance)

        if not user_utterance:
            raise Exception("Invalid user_utterance.")

        try:
            model = self._get_model()
            result = model.classify(user_utterance)

            if is_debug():
                print("ClassifyStage1 result", result)

            return self.create_text_message(json.dumps(result, ensure_ascii=False))
        except Exception as e:
            if is_debug():
                import traceback
                print(traceback.format_exc())
            raise Exception(f"Failed to classify stage1: {e}")


if __name__ == "__main__":
    tool = ClassifyStage1Tool()
    sample = sys.argv[1] if len(sys.argv) > 1 else "생일 혜택 뭐 있어?"
    print(tool._invoke(user_id="poc", tool_parameters={"user_utterance": sample}))
