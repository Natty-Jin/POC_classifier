"""
2단계 분류기 학습: 멤버십 혜택 세부 8클래스

주의: 이 학습 데이터는 "1단계에서 membership으로 분류될 발화"만 대상으로
라벨링해야 합니다. knowledge에 해당하는 발화는 stage2 데이터에 포함하지 마세요.

실행:
    python train_stage2.py
"""

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
DATA_PATH = os.path.join(ROOT_DIR, "data", "stage2_labeled.xlsx")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "stage2_classifier.joblib")

VALID_LABELS = {
    "vip_choice",
    "multi_condition",
    "point_abolition",
    "grade_info",
    "birthday_benefit",
    "dal_benefit",
    "membership_type",
    "usage_history",
}


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} 없음. 먼저 data/generate_label_template.py 실행 후 라벨링하세요."
        )

    df = pd.read_excel(DATA_PATH, sheet_name="stage2_라벨링", header=2)
    df = df.dropna(subset=["utterance", "label"])
    df = df[df["labeler"] != "예시"]

    if len(df) == 0:
        raise ValueError("라벨링된 데이터가 없습니다. 엑셀에 발화와 라벨을 채워주세요.")

    invalid = df[~df["label"].isin(VALID_LABELS)]
    if len(invalid) > 0:
        print("잘못된 라벨 발견:")
        print(invalid[["id", "utterance", "label"]])
        raise ValueError(f"label 값은 다음 중 하나여야 합니다: {sorted(VALID_LABELS)}")

    print("클래스별 데이터 개수:")
    counts = df["label"].value_counts()
    print(counts)
    print()

    missing = VALID_LABELS - set(counts.index)
    if missing:
        print(f"⚠ 경고: 아직 데이터가 하나도 없는 클래스: {sorted(missing)}")
    if counts.min() < 5:
        print("⚠ 경고: 일부 클래스 데이터가 5개 미만입니다. 검증 결과가 불안정할 수 있습니다.\n")

    X_train, X_val, y_train, y_val = train_test_split(
        df["utterance"], df["label"],
        test_size=0.2, stratify=df["label"], random_state=42,
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)
    print("=== 검증 리포트 ===")
    print(classification_report(y_val, preds))
    print("=== Confusion Matrix (rows=실제, cols=예측) ===")
    labels_sorted = sorted(VALID_LABELS)
    cm = confusion_matrix(y_val, preds, labels=labels_sorted)
    print(pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted))
    print(
        "\n힌트: 대각선 밖에 값이 몰려있는 행/열 쌍이 실제로 헷갈리는 카테고리입니다. "
        "해당 페어 데이터를 더 채우거나 라벨링 가이드를 다듬으세요."
    )

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n저장 완료: {MODEL_PATH}")


if __name__ == "__main__":
    main()
