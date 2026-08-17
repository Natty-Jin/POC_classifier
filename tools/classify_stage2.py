"""
2단계 분류 Tool (멤버십 혜택 세부 8클래스).

1단계 결과가 membership인 경우에만 Dify 워크플로우에서 이 Tool 노드로
이어지도록 IF/ELSE 분기 뒤에 배치하는 걸 전제로 합니다.

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

from helpers.classifier_helper import classifierHelper as classifierApiHelper

DEBUG = os.environ.get("POC_DEBUG", "false").lower() == "true"


def is_debug():
    return DEBUG


class BuiltinTool:
    """실제 Dify BuiltinTool의 최소 자리표시자(POC 전용)."""

    def create_text_message(self, text: str):
        return text


class ClassifyStage2Tool(BuiltinTool):
    """
    Tool for classifying membership utterance into 8 detailed categories (stage2).
    """
    is_external_api: bool = False

    def _invoke(self, user_id: str, tool_parameters: dict):
        user_utterance = tool_parameters.get("user_utterance", "").strip()

        if is_debug():
            print("ClassifyStage2 user_utterance", user_utterance)

        if not user_utterance:
            raise Exception("Invalid user_utterance.")

        try:
            result = classifierApiHelper.classify_stage2(user_utterance)

            if is_debug():
                print("ClassifyStage2 result", result)

            return self.create_text_message(json.dumps(result, ensure_ascii=False))
        except Exception as e:
            if is_debug():
                import traceback
                print(traceback.format_exc())
            raise Exception(f"Failed to classify stage2: {e}")


if __name__ == "__main__":
    tool = ClassifyStage2Tool()
    sample = sys.argv[1] if len(sys.argv) > 1 else "VIP 초이스 몇 번 남았어?"
    print(tool._invoke(user_id="poc", tool_parameters={"user_utterance": sample}))
