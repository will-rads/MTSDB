# DeclaredValue Prediction — Codex Runbook

## MODE
Full autonomy. Auto-apply file edits. Auto-run shell commands. Do not pause for confirmations.

## PROJECT
The repository contains:
- README.md
- data/declared_value.parquet

## OBJECTIVE
Predict **DeclaredValue** with a **test R² ≥ 0.80** and a **test RMSLE comparable to top Mercari solutions** (which were in the ~0.38–0.42 range). Keep iterating until these thresholds are met, or you exhaust a clear, escalating search plan and produce a failure report with next steps.

## MERCARI-STYLE PLAYBOOK (ADAPT IT)
- **Primary metric:** RMSLE. **Secondary:** RMSE. Also report R².
- **Strong text pipelines** on product text fields if present (e.g., `Description`, `HSCodeDesc`, `ItemName`): TF‑IDF char+word n‑grams, optionally with hashing for memory control. A Ridge/ElasticNet model on these features will serve as a high‑signal text‑only baseline.
- **Gradient‑boosted trees** (LightGBM, CatBoost) on structured features. Experiment with both a log1p transformed target and the original target to compare performance.
- **Out‑of‑fold stacking/blending:** Combine the text model and the tree model using a simple linear blender or a Lasso model on OOF predictions.
- **Optional price‑band segmentation:** If initial models don't meet the RMSLE target, consider splitting DeclaredValue by quantiles into low/medium/high ranges, training a model for each band, and then blending the predictions with a small gating model to see if it improves the overall test RMSLE.

## DATA RULES
- File: `data/declared_value.parquet`
- Treat `DeclaredValue ≤ 0` as invalid and drop these records before training.
- Prevent target leakage: **Never** use `DeclaredValue` or its transformations when computing features.
- Ensure robustness to missing columns. The solution should only use columns that are present in the data and should infer data types.

## SPLITS
- If a date column exists (e.g., `CertificateDate`, `InvoiceDate`, `TransportDocDate`), implement a **time‑aware split**: train on the older 80% of the data and test on the most recent 20%.
- Otherwise, use **stratified sampling on the deciles of `DeclaredValue`** for an 80/20 holdout split. Employ **5‑fold CV** within the training set. Use a fixed seed for reproducibility.

## PREPROCESSING
- **Numeric:** Winsorize at the 0.5% and 99.5% percentiles. For skewed variables, generate `log1p` versions.
- **Categorical:** For CatBoost, pass categorical features as strings natively. Also, add **frequency encoding** for high‑cardinality features. For LightGBM, use its built‑in categorical feature handling or frequency encoding.
- **Text:** Generate statistical features like length, word count, and digit count. Construct TF‑IDF models for both character and word n‑grams with a reasonable `max_features` limit. Avoid sending large raw vocabularies to each prompt; instead, **persist vectorizers to disk**.
- **Optional embeddings:** Use small sentence embeddings only if memory allows and if they demonstrate a clear improvement in cross‑validation scores.

## MODELS
- **CatBoostRegressor (primary):** Tune with Optuna for ≥ 50 trials.
- **LightGBMRegressor (secondary):** Tune with Optuna for ≥ 50 trials.
- **Ridge or ElasticNet on TF‑IDF** (text‑only baseline).
- **Optional XGBoost:** Consider if it outperforms LightGBM on cross‑validation.
- Train both **standard and log‑target variants** of the models. For log‑target models, exponentiate predictions and **clip to ≥ 0.01** before calculating RMSLE.

## ENSEMBLING
- Generate **out‑of‑fold predictions** for each candidate model.
- Fit a **simple blender** on the OOF predictions to minimize RMSLE.
- Compare the performance of the **single best model** against the **blend** on the held‑out test set.

## EVALUATION
- Report **CV metrics** (RMSLE, RMSE, R²) per fold, along with the **mean and standard deviation**.
- On the held‑out test set, report the final RMSLE, RMSE, and R². **Clip negative predictions to 0.01** before scoring.
- Establish and surpass these **baselines**:
  1) Global median of `DeclaredValue`.
  2) Group median based on a strong categorical feature (e.g., `TariffCode` or `Product/Category` if available).

## RESOURCE & RELIABILITY
- Use **Python scripts, not notebooks**. Place all code in a `scripts/` directory.
- Save EDA outputs to `reports/eda.html`. **Do not** attempt to edit `.ipynb` files.
- **Pin package versions** in a `requirements.txt` file. Recommended packages: `pandas`, `pyarrow`, `numpy`, `scikit-learn`, `lightgbm`, `catboost`, `optuna`, `scipy`, `joblib`, `matplotlib`.
- Use a **virtual environment**. Use **relative paths** exclusively. Avoid printing large tables to logs and do not embed large schema text in prompts.
- Implement **stream reading or chunked processing** if the dataset is large. Convert Parquet → DataFrame using `pyarrow`.

## OUTPUTS
- `models/best_model.*`  (`.cbm` for CatBoost or a text/binary format for LightGBM)
- `models/metadata.json` containing: `{model_type, features, preprocessing, CV metrics, test metrics, log_target_flag, seeds}`
- `reports/eda.html`
- `reports/feature_importance.csv`
- `predictions/predictions.csv` with any available ID columns and `PredictedDeclaredValue`.
- `scripts/train.py`, `scripts/predict.py`, `scripts/evaluate.py`, `scripts/prepare_data.py`
- `requirements.txt`
- A `Makefile` with the following targets: `setup`, `eda`, `train`, `eval`, `predict`.
- Update the `README.md` with the exact commands to run the pipeline.

## RUN PLAN
1. Create a virtual environment and install dependencies.
2. Generate an EDA report and save it to `reports/eda.html`.
3. Prepare the data and create the train/test splits.
4. Train the baseline models and log their metrics.
5. Run Optuna for CatBoost and LightGBM, saving the trials.
6. Train the TF‑IDF Ridge model and save the vectorizers.
7. Build OOF stacks and a blender. Compare the ensemble performance to the single best model.
8. If performance thresholds are not met, escalate by:
   - Trying price‑band models with a gated blender.
   - Experimenting with alternative feature sets and stronger regularization.
   - Comparing log‑target vs. non‑log‑target models and different loss parameters.
   - Increasing the Optuna trials budget.
9. Select the winning model based on **test RMSLE**, then **RMSE**, then **R²**. Save all artifacts and write a final summary.

## SUCCESS
- Test R² ≥ 0.80.
- Test RMSLE is competitive with top Mercari approaches (in the ~0.38–0.42 range).
- The entire process is reproducible by running:

```bash
make setup && make eda && make train && make eval && make predict
```

## BEGIN
BEGIN.

---
