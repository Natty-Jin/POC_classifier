"""
학습된 모델로 발화 하나를 1단계 → (필요 시) 2단계 순서로 분류해보는 데모.

실행:
    python infer_demo.py "생일 혜택 뭐 있어?"
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.classifier_helper import classifierHelper


def classify(utterance: str) -> dict:
    stage1_result = classifierHelper.classify_stage1(utterance)
    result = {"utterance": utterance, "stage1": stage1_result}

    if stage1_result["category"] == "membership":
        stage2_result = classifierHelper.classify_stage2(utterance)
        result["stage2"] = stage2_result

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('사용법: python infer_demo.py "분류할 발화"')
        sys.exit(1)

    utterance = sys.argv[1]
    result = classify(utterance)
    print(json.dumps(result, ensure_ascii=False, indent=2))
