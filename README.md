# Freight Rate Prediction

Predict total posted load rates with chronological evaluation and histogram gradient boosting. The selected main model achieved September-October holdout MAE of **$120.17**, RMSE of **$638.06**, and WAPE of **5.02%**. Hidden November-December outcomes are not available.

## Files

- `Freight_Rate_Prediction.ipynb`: cleaned notebook with explanatory sections; run in order.
- `run.py`: equivalent noninteractive command-line workflow.
- `requirements.txt`: training and scoring dependencies.
- `score.py`: original supplied output validator and chart generator.
- `validation_predictions.csv`: 12,000 final load predictions.
- `december_predictions.csv`: 31 completed fixed-scenario rows.
- `scorer_results/candidate_december.png`: original scorer chart.
- `Freight_Rate_Report.pdf`: methodology, validation, results and limitations.
- `results/`: original Colab evaluation evidence and recorded package versions.
- `LOOM_GUIDE.md`: suggested 2-3 minute walkthrough.

## Run locally

Use Python 3.11 or 3.12. From the extracted repository directory:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py
```

First place the four original assessment CSVs in `data/`:

| Input | Accepted filenames |
| --- | --- |
| Labeled data | `train-test.csv` or `train_test.csv` |
| Final load inputs | `validation.csv` or `validation - Copy.csv` |
| Prediction template | `validation-predictions-template.csv` or `validation_predictions_template.csv` |
| December inputs | `december-chart-inputs.csv` or `december_chart_inputs.csv` |

Inputs beside the notebook/script are also supported. Raw assessment inputs are not included in this package. Outputs already included allow the scorer to run without the training CSVs.

`run.py` reruns model selection, evaluates the locked models, refits on all labeled rows, and overwrites the two prediction files, metrics and chart. The original December input file is not changed.

For the exact originally recorded package versions, install `results/environment_versions.txt`. The ordinary requirements file pins scikit-learn and allows compatible numerical/plotting dependency versions. The script was independently rerun and produced identical rounded prediction CSVs to the original Colab run.

## Run in Colab

Upload `Freight_Rate_Prediction.ipynb` to Colab. Run its first upload cell and select the four original CSV files plus this package's `requirements.txt` and `score.py`. Run the remaining cells in order with a CPU runtime. Installing the requirements in a fresh runtime before importing the libraries is recommended. Raw uploads and generated files must be downloaded before a temporary Colab runtime is lost.

## Validate included outputs

```bash
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

Expected: 12,000 final predictions and 31 December predictions validated; `scorer_results/candidate_december.png` created. The scorer checks format and values, not predictive accuracy. Employer-side metrics are calculated after submission.

## Method

1. Treat non-positive weight as missing; retain all rows and target extremes.
2. Train candidate models on January-June (28,806 rows) and select by July-August MAE (9,671 rows).
3. Fit category encoders only on the applicable training period. Unknown categories map to missing values; numeric missing values are handled by the estimator.
4. Select the main model without `quote_signal`: MAE 137.34 vs 139.65 with it. A reduced-input December model has selection MAE 171.15.
5. Lock both configurations, train on January-August (38,477 rows), and evaluate September-October (9,523 rows).
6. Refit on all 48,000 labeled rows; match final predictions to template load IDs.

The primary model predicts total dollars, uses absolute-error loss, 250 iterations, learning rate 0.06, at most 23 leaves, minimum 45 rows per leaf, L2 regularization 2.0, no internal early stopping, and seed 42. Calendar features do not establish full annual seasonality. The December model excludes coordinates, market_index and quote_signal because those fields are unavailable in the scenario input.

The reduced-input model achieves lower holdout MAE ($111.63), but the July-August primary choice is retained rather than reselecting on holdout results. WAPE is an error ratio, not accuracy. See the report for remaining uncertainty and limitations.

## Submission

Upload the report to the report field. Include the prediction CSVs and chart in the repository. Provide a repository link accessible to reviewers and a recorded 2-3 minute Loom link. This package does not publish a repository or record a video. Keep original assessment data private unless the employer permits publication; arrange reviewer access if the repository is private.
