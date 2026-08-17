"""
1단계 분류 Tool (지식 안내 / 멤버십 혜택 안내).

이 파일은 Dify 레포 밖(POC)에서 구조를 미리 잡아두기 위한 스텁입니다.
실제 Dify 레포로 이관 시 아래 "실제 이관 시" 주석의 import로 교체하고,
BASE 클래스를 core.tools.tool.builtin_tool.BuiltinTool 로 바꾸면 그대로 동작합니다.

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


class ClassifyStage1Tool(BuiltinTool):
    """
    Tool for classifying user utterance into knowledge/membership (stage1).
    """
    is_external_api: bool = False

    def _invoke(self, user_id: str, tool_parameters: dict):
        user_utterance = tool_parameters.get("user_utterance", "").strip()

        if is_debug():
            print("ClassifyStage1 user_utterance", user_utterance)

        if not user_utterance:
            raise Exception("Invalid user_utterance.")

        try:
            result = classifierApiHelper.classify_stage1(user_utterance)

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
