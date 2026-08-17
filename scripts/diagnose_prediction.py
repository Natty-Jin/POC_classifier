"""
특정 문장이 왜 knowledge/membership으로 분류됐는지, 어떤 글자 조각(n-gram)이
얼마나 영향을 줬는지 상위 기여도 순으로 보여주는 진단 도구.

실행:
    python diagnose_prediction.py "이번달 내가 쓴 내역 뭐야?"
"""

import os
import sys
import joblib
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
MODEL_PATH = os.path.join(ROOT_DIR, "models", "stage1_classifier.joblib")


def diagnose(text: str, top_n: int = 15):
    pipeline = joblib.load(MODEL_PATH)
    vectorizer = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    x = vectorizer.transform([text]).toarray()[0]  # TF-IDF 벡터
    coef = clf.coef_[0]  # membership(양성 클래스) 기준 가중치
    contribution = x * coef  # 각 n-gram이 최종 점수에 기여한 양

    feature_names = vectorizer.get_feature_names_out()
    nonzero_idx = np.nonzero(x)[0]

    rows = [(feature_names[i], x[i], coef[i], contribution[i]) for i in nonzero_idx]
    rows.sort(key=lambda r: -abs(r[3]))

    proba = pipeline.predict_proba([text])[0]
    classes = clf.classes_
    pred = classes[np.argmax(proba)]

    print(f'문장: "{text}"')
    print(f"예측: {pred}  (knowledge={proba[0]:.4f}, membership={proba[1]:.4f})")
    print(f"\n이 문장에 등장한 n-gram 중, 예측에 영향을 준 상위 {top_n}개")
    print("(양수 = membership 쪽으로 끌어당김, 음수 = knowledge 쪽으로 끌어당김)\n")
    print(f'{"n-gram":10s} {"tfidf값":>10s} {"학습가중치":>12s} {"기여도":>10s}')
    for token, tfidf_val, weight, contrib in rows[:top_n]:
        direction = "→membership" if contrib > 0 else "→knowledge "
        print(
            f"{repr(token):10s} {tfidf_val:10.4f} {weight:12.4f} {contrib:10.4f}  {direction}"
        )

    total = contribution.sum() + clf.intercept_[0]
    print(f"\n전체 n-gram 기여도 합 + intercept = {total:.4f} (양수면 membership 우세)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('사용법: python diagnose_prediction.py "분석할 문장"')
        sys.exit(1)
    diagnose(sys.argv[1])
