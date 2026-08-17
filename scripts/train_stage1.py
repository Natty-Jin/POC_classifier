"""
1단계 분류기 학습: 지식 안내(knowledge) vs 멤버십 혜택 안내(membership)

실행:
    python train_stage1.py
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
DATA_PATH = os.path.join(ROOT_DIR, "data", "stage1_labeled.xlsx")
MODEL_PATH = os.path.join(ROOT_DIR, "models", "stage1_classifier.joblib")
VALID_LABELS = {"knowledge", "membership"}


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"{DATA_PATH} 없음. 먼저 data/generate_label_template.py 실행 후 라벨링하세요."
        )

    df = pd.read_excel(DATA_PATH, sheet_name="stage1_라벨링", header=2)
    df = df.dropna(subset=["utterance", "label"])
    df = df[df["labeler"] != "예시"]  # 템플릿 예시 행 제외

    if len(df) == 0:
        raise ValueError("라벨링된 데이터가 없습니다. 엑셀에 발화와 라벨을 채워주세요.")

    invalid = df[~df["label"].isin(VALID_LABELS)]
    if len(invalid) > 0:
        print("잘못된 라벨 발견:")
        print(invalid[["id", "utterance", "label"]])
        raise ValueError("label 값은 knowledge 또는 membership만 허용됩니다.")

    print("클래스별 데이터 개수:")
    print(df["label"].value_counts())
    print()

    if df["label"].value_counts().min() < 5:
        print("⚠경고: 일부 클래스 데이터가 5개 미만입니다. 검증 결과가 불안정할 수 있습니다.\n")

    X_train, X_val, y_train, y_val = train_test_split(
        df["utterance"], df["label"],
        test_size=0.2, stratify=df["label"], random_state=42,
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_val)
    print("=== 검증 리포트 ===")
    print(classification_report(y_val, preds))
    print("=== Confusion Matrix (rows=실제, cols=예측) ===")
    labels_sorted = sorted(VALID_LABELS)
    print(pd.DataFrame(
        confusion_matrix(y_val, preds, labels=labels_sorted),
        index=labels_sorted, columns=labels_sorted,
    ))

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n저장 완료: {MODEL_PATH}")


if __name__ == "__main__":
    main()
