"""
scikit-learn으로 학습된 파이프라인(TfidfVectorizer + LogisticRegression)의
학습 결과(단어사전, IDF, 회귀계수)만 뽑아서 순수 JSON으로 저장합니다.

이 JSON은 상용 서빙 환경에서 scikit-learn/pandas/joblib 없이,
numpy만으로 그대로 읽어서 추론할 수 있습니다.

실행:
    python export_to_numpy.py stage1
"""

import os
import sys
import json
import joblib
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")


def export(stage_name: str):
    joblib_path = os.path.join(MODELS_DIR, f"{stage_name}_classifier.joblib")
    json_path = os.path.join(MODELS_DIR, f"{stage_name}_classifier_numpy.json")

    if not os.path.exists(joblib_path):
        raise FileNotFoundError(f"{joblib_path} 없음. 먼저 학습을 실행하세요.")

    pipeline = joblib.load(joblib_path)
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    if vectorizer.analyzer != "char_wb":
        raise ValueError("이 export 스크립트는 analyzer='char_wb' 전제입니다.")

    artifact = {
        "analyzer": "char_wb",
        "ngram_range": list(vectorizer.ngram_range),
        "vocabulary": vectorizer.vocabulary_,          # {token: index}
        "idf": vectorizer.idf_.tolist(),                # len == vocab size
        "classes": clf.classes_.tolist(),                # ['knowledge', 'membership']
        "coef": clf.coef_.tolist(),                      # shape (1, vocab_size) for binary
        "intercept": clf.intercept_.tolist(),             # shape (1,)
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, ensure_ascii=False)

    size_kb = os.path.getsize(json_path) / 1024
    print(f"저장됨: {json_path} ({size_kb:.1f} KB, vocab {len(vectorizer.vocabulary_)}개)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("사용법: python export_to_numpy.py <stage1|stage2>")
        sys.exit(1)
    export(sys.argv[1])
