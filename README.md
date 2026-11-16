dsi_final_project/
├── README.md # this file
├── requirements.txt # dependencies
├── Dockerfile # Docker build
├── railway.toml # Railway build + deploy config
├── start.sh # local start script
├── notebook/
│ └── final_project.ipynb # ALL phases 1-6 in ONE notebook — run top to bottom
├── app/
│ ├── app.py # Phase 6 — FastAPI app
│ ├── index.html # Phase 6 — Single + Batch UI
│ └── data/ # app data if needed
├── model/
│ ├── final_model.pkl # XGBoost v1, depth 4, n 200 (= model_v1_xgboost_depth4_n200.pkl)
│ ├── scaler.pkl
│ └── feature_columns.pkl
└── outputs/ # if present: metrics.csv + plots
├── metrics.csv
├── roc.png
├── model_comparison.png
└── sentiment.png


jupiter notebook

Open ONE notebook — notebook/final_project.ipynb — and click Run All from top to bottom.
Do not run cells out of order. Phases inside the notebook are in this order:

Phase 1 — Cleaning + Integration
Fix invalid dates, Response Yes/No → 1/0, numeric coercion, log1p(total_sales), join Transaction aggregates to Master. No leakage: no future Campaign info in features.

Phase 2 — EDA
Box-whisker NPS / n_comp / n_communications by Response, response-rate tables Loyalty + NPS, bar charts Features 1/2/3, MoM sales, top products. Overall Response Rate 56.8%.

Phase 3 — Logistic Regression (11 steps)

    Split 80/20 stratified
    Logit with all variables
    Significance p < 0.05
    Relate to EDA (signs only)
    VIF — total_sales_log 12.23, n_communications 6.64, retained: removal did not improve Test AUC
    ROC + AUC Train 0.6405
    Confusion Matrix Train — Accuracy, Sensitivity, Specificity, Balanced Accuracy via get_metrics()
    Threshold to balance Sens/Spec — Youden 0.565, Sens 0.66, Spec 0.59
    Go to Test ONLY if satisfied with Train — ROC + AUC Test 0.6225, gap 0.018
    Test Sens/Spec at same threshold
    Version 2 with recoded variables — finalise on Test AUC only

Phase 4 — Model Comparison
Compare Logistic, Naive Bayes, Random Forest, SVM, XGBoost. Select by max Test AUC. Tuning: max_depth 3/4/6, n_estimators 200 vs 300. Rejected max_depth=6 (Test 0.7493, gap 0.189, overfit) and n_estimators=300 (Test 0.7220, gap 0.159). Final XGBoost depth 4, n 200.

Phase 5 — Sentiment
VADER on 965 reviews, WordCloud, bigrams. No labels → no Accuracy. Results: 63.11% / 24.56% / 12.33%.

Phase 6 — Deployment
Notebook saves model + scaler + feature_columns to model/. API code itself is in app/app.py — see section 3.3.



### How to run  project

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# from file app/
uvicorn app:app --reload --port 8000


Docker (optional)
docker build -t dsi-final .
docker run -p 8000:8000 -e PORT=8000 dsi-final

git
https://github.com/MykolaIvanowski/dsi_final_project/