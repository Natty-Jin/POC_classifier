# venv 확인용
"C:\Users\????????\.pyenv\pyenv-win\versions\3.10.11\python.exe" -m venv .venv

파이썬 venv 가상 환경 해키지 관리 디렉토리는 기존에 설정한 path가 있기 때문에 라이브러리 설치가 안되는 경우가 허다함. 이럴 경우 pyenv local {원하는 버전}으로 변경해보거나 pyenv version 또는 pyenv versions를 터미널에서 확인한 후 안된다 싶으면 맨 위 명령어로 .venv 폴더를 만든다.

#1. 학습 먼저
python scripts\train_stage1.py


## 학습 이후, models 폴더에 stage1_classifier.joblib가 생김

#2. 인자 넘기기 
python scripts\export_to_numpy.py stage1

## stage1 numpy 연산 후, json 파일이 models에 생김. `stage1_classifier_numpy.json`

#3. numpy tool 실행
python tools\classify_stage1_numpy.py "생일 혜택 뭐 있어?"

# 발화 분류기 POC (지식 안내 / 멤버십 혜택 안내 → 세부 카테고리)

TF-IDF + LogisticRegression 기반 경량 2단계 발화 분류기. 딥러닝 프레임워크(torch/transformers/onnxruntime) 없이 `scikit-learn`만으로 동작하도록 설계했습니다.

## 구조

```
poc-classifier/
├── data/
│   ├── stage1_labeled.xlsx      # 1단계 라벨링 데이터 (지식/멤버십)
│   ├── stage2_labeled.xlsx      # 2단계 라벨링 데이터 (멤버십 세부 8클래스)
│   └── generate_label_template.py
├── models/                      # 학습된 .joblib 결과물이 저장되는 곳 (실행 전엔 비어있음)
├── helpers/
│   └── classifier_helper.py     # 두 모델을 로드/관리하는 헬퍼 (추후 core/helper/agent_api_helper/ 위치로 그대로 이동)
├── tools/
│   ├── classify_stage1.py       # Dify BuiltinTool 형태 스텁 (1단계)
│   └── classify_stage2.py       # Dify BuiltinTool 형태 스텁 (2단계)
├── scripts/
│   ├── train_stage1.py
│   ├── train_stage2.py
│   └── infer_demo.py            # 학습 없이 바로 결과 확인용 CLI 데모
└── requirements.txt
```

## 사용 순서

1. `pip install -r requirements.txt`
2. `python data/generate_label_template.py` → `data/stage1_labeled.xlsx`, `data/stage2_labeled.xlsx` 생성 (예시 행 포함, 드롭다운 라벨 검증 걸려있음)
3. 엑셀 열어서 `utterance` 채우고 `label` 드롭다운으로 라벨링
   - stage1: `knowledge` / `membership`
   - stage2: **`membership`으로 분류된 발화만** 대상으로, 8개 세부 카테고리 중 하나
4. `python scripts/train_stage1.py` → `models/stage1_classifier.joblib` 생성 + 검증 리포트(classification_report, confusion matrix) 출력
5. `python scripts/train_stage2.py` → `models/stage2_classifier.joblib` 생성 + 검증 리포트 출력
6. `python scripts/infer_demo.py "생일 혜택 뭐 있어?"` → 1단계→2단계 순차 분류 결과 확인

## 카테고리 정의 (2단계, 8클래스)

| 코드 | 설명 |
|---|---|
| `vip_choice` | VIP 초이스 혜택 (이력, 사용여부, 잔여 횟수) |
| `multi_condition` | 다중조건혜택안내 (업종별/브랜드별/생일 등 조건 결합) |
| `point_abolition` | 포인트 폐지 |
| `grade_info` | 사용자 등급 안내 |
| `birthday_benefit` | 생일 혜택 안내 (단독) |
| `dal_benefit` | 달달 혜택 (과거/현재) |
| `membership_type` | 멤버십 종류별 혜택 안내 |
| `usage_history` | 사용이력안내 |

세부 조건(괄호 안 내용 — 특정 브랜드명, 기간, 잔여 횟수 등)은 이 분류기가 아니라 이후 LLM 노드에서 재분류합니다.

## 추후 파일 저장소(Dify 레포) 이관 시

- `helpers/classifier_helper.py` → `core/helper/agent_api_helper/classifier_api_helper.py` 로 경로만 옮기면 그대로 동작 (import 문의 상대경로 수정 필요)
- `tools/classify_stage1.py`, `tools/classify_stage2.py` → `core/tools/provider/builtin/category_classifier/tools/` 로 이동, 기존 BuiltinTool 임포트 경로(`core.tools.entities...` 등)로 교체
- `models/*.joblib` → `models/category_classifier/` 로 이동, 헬퍼의 경로 상수만 맞추면 됨
- 이 시점에 실제 프로젝트의 `is_debug()`, `BuiltinToolApiError` 등을 다시 연결


## 에러 발생 시 아래 내용 참고 ↓ 
## 추가로 3.9 ~ 3.13버전 사이만 scikit-learn 1.5.1용 wheel을 사용 할 수 있음. 
- python -m pip install --upgrade pip setuptools wheel