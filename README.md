# Cancer Risk Prediction

This project analyzes a public cancer-risk dataset, trains a model that predicts **Low / Medium / High** risk, and serves that model through a REST API.

The model is for research and education. It is **not** a medical diagnosis.

## Dataset

`cancer data.csv` has 1,000 patients and no missing values. Each row includes age, gender, lifestyle and exposure scores, symptoms, and a `Level` target.

Features used by the model:

- Age, Gender
- Air Pollution, Alcohol use, Dust Allergy, OccuPational Hazards
- Genetic Risk, chronic Lung Disease, Balanced Diet, Obesity
- Smoking, Passive Smoker
- Chest Pain, Coughing of Blood, Fatigue, Weight Loss
- Shortness of Breath, Wheezing, Swallowing Difficulty
- Clubbing of Finger Nails, Frequent Cold, Dry Cough, Snoring

`index` and `Patient Id` are identifiers and are not used for training.

## Setup

Use Python 3.13 (the project venv is created with it):

```bash
py -3.13 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Analyze

Exploratory analysis is in the notebook:

```bash
jupyter notebook eda.ipynb
```

It covers data quality, class balance, age/gender, risk-factor boxplots, and correlations, then writes:

- `figures/01_target_distribution.png`
- `figures/02_age_gender.png`
- `figures/03_correlation_heatmap.png`
- `figures/04_feature_boxplots.png`

`eda.py` only holds shared helpers used by the notebook and `train.py`.

## Train

```bash
python train.py
```

`train.py` compares logistic regression and random forest, keeps the forest when scores tie, and saves:

- `models/cancer_risk_model.joblib`
- `models/metrics.json`
- `figures/05_confusion_matrix.png`
- `figures/06_feature_importance.png`

On the held-out test set both models reach accuracy 1.0 and macro F1 1.0. That happens because the 1,000 rows contain only 152 unique feature profiles, and each profile has one label. The API is still useful as a demo; it is not a clinical diagnostic tool.

## API

Start the server after training:

```bash
uvicorn app:app --reload
```

Interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/health` | Service status |
| GET | `/features` | Required input fields |
| GET | `/metrics` | Held-out test metrics |
| POST | `/predict` | Predict risk level |

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"Age\": 35, \"Gender\": 1, \"Air Pollution\": 4, \"Alcohol use\": 5, \"Dust Allergy\": 6, \"OccuPational Hazards\": 5, \"Genetic Risk\": 5, \"chronic Lung Disease\": 4, \"Balanced Diet\": 6, \"Obesity\": 7, \"Smoking\": 2, \"Passive Smoker\": 3, \"Chest Pain\": 4, \"Coughing of Blood\": 8, \"Fatigue\": 8, \"Weight Loss\": 7, \"Shortness of Breath\": 9, \"Wheezing\": 2, \"Swallowing Difficulty\": 1, \"Clubbing of Finger Nails\": 4, \"Frequent Cold\": 6, \"Dry Cough\": 7, \"Snoring\": 2}"
```

Example response:

```json
{
  "prediction": "High",
  "probabilities": {
    "Low": 0.01,
    "Medium": 0.04,
    "High": 0.95
  },
  "model_name": "random_forest"
}
```

Use the original CSV column names in the JSON body. `Gender` must be `1` or `2`.
