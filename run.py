"""Reproduce the cleaned notebook from the repository root."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
from pathlib import Path
import pandas as pd
def display(value):
    print(value.to_string() if hasattr(value, "to_string") else value)

# Inputs may be in data/ or beside this notebook.
def input_path(*names):
    for folder in [Path("data"), Path(".")]:
        for name in names:
            path = folder / name
            if path.is_file():
                return path
    raise FileNotFoundError(f"Missing input: {names}. Upload the supplied assessment files.")
import pandas as pd

# Load the four datasets
train_df = pd.read_csv(input_path("train-test.csv", "train_test.csv"))
validation_df = pd.read_csv(input_path("validation.csv", "validation - Copy.csv"))
template_df = pd.read_csv(input_path("validation-predictions-template.csv", "validation_predictions_template.csv"))
december_df = pd.read_csv(input_path("december-chart-inputs.csv", "december_chart_inputs.csv"))

# Display the number of rows and columns
datasets = {
    "Training data": train_df,
    "Validation data": validation_df,
    "Prediction template": template_df,
    "December inputs": december_df,
}

for name, df in datasets.items():
    print(f"{name}: {df.shape[0]:,} rows, {df.shape[1]} columns")

# Preview training examples
print("\nFirst five training rows:")
display(train_df.head())

# Inspect column types and non-missing values
print("\nTraining data information:")
train_df.info()


# Check missing values in training and final prediction data
missing_summary = pd.DataFrame({
    "Training missing": train_df.isna().sum(),
    "Validation missing": validation_df.isna().sum()
}).fillna("Not present")

print("Missing values:")
display(missing_summary)

# Check duplicate rows and shipment IDs
for name, df in [
    ("Training", train_df),
    ("Validation", validation_df)
]:
    print(f"\n{name}:")
    print("Duplicate rows:", df.duplicated().sum())
    print("Duplicate load IDs:", df["load_id"].duplicated().sum())
    print("Non-positive weights:", (df["weight"] <= 0).sum())
    print("Non-positive distances:", (df["distance"] <= 0).sum())

print(
    "\nNon-positive training rates:",
    (train_df["posted_rate"] <= 0).sum()
)

# Inspect dates without changing the original columns
for name, df in [
    ("Training", train_df),
    ("Validation", validation_df),
    ("December", december_df)
]:
    parsed_dates = pd.to_datetime(
        df["date"], format="%Y-%m-%d", errors="coerce"
    )

    print(f"\n{name} dates:")
    print("Invalid or missing dates:", parsed_dates.isna().sum())
    print("First date:", parsed_dates.min())
    print("Last date:", parsed_dates.max())

# Inspect numeric ranges and possible extreme values
print("\nTraining numeric summary:")
display(
    train_df[
        ["distance", "weight", "market_index", "quote_signal", "posted_rate"]
    ].describe(percentiles=[0.01, 0.50, 0.99]).round(2)
)


import numpy as np

# Create working copies and preserve the original data
train_clean = train_df.copy()
validation_clean = validation_df.copy()
december_clean = december_df.copy()

# Parse dates and mark invalid weights as missing
for df in [train_clean, validation_clean, december_clean]:
    df["date"] = pd.to_datetime(
        df["date"], format="%Y-%m-%d", errors="raise"
    )
    df.loc[df["weight"] <= 0, "weight"] = np.nan

# Split labeled data chronologically
fit_df = train_clean.loc[
    train_clean["date"] < "2025-07-01"
].copy()

tune_df = train_clean.loc[
    (train_clean["date"] >= "2025-07-01")
    & (train_clean["date"] < "2025-09-01")
].copy()

holdout_df = train_clean.loc[
    train_clean["date"] >= "2025-09-01"
].copy()

# Verify that every labeled row belongs to one split
assert len(fit_df) + len(tune_df) + len(holdout_df) == len(train_clean)
assert fit_df["date"].max() < tune_df["date"].min()
assert tune_df["date"].max() < holdout_df["date"].min()

for name, df in [
    ("Initial training", fit_df),
    ("Model selection", tune_df),
    ("Final holdout", holdout_df),
]:
    print(
        f"{name}: {len(df):,} rows | "
        f"{df['date'].min():%Y-%m-%d} to "
        f"{df['date'].max():%Y-%m-%d}"
    )

print("\nMissing weights after cleaning:")
print("Training:", train_clean["weight"].isna().sum())
print("Validation:", validation_clean["weight"].isna().sum())

# Learn a simple rate per mile from training data only
baseline_rate_per_mile = (
    fit_df["posted_rate"] / fit_df["distance"]
).median()

# Predict July and August prices
baseline_predictions = (
    tune_df["distance"] * baseline_rate_per_mile
)

# Define reusable evaluation metrics
def evaluate_predictions(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = actual - predicted

    return {
        "MAE ($)": np.mean(np.abs(errors)),
        "RMSE ($)": np.sqrt(np.mean(errors ** 2)),
        "WAPE (%)": (
            100 * np.sum(np.abs(errors)) / np.sum(np.abs(actual))
        ),
    }

baseline_metrics = evaluate_predictions(
    tune_df["posted_rate"],
    baseline_predictions
)

print(f"Baseline rate per mile: ${baseline_rate_per_mile:.4f}")
print("\nJuly-August evaluation:")

for name, value in baseline_metrics.items():
    print(f"{name}: {value:,.2f}")

# Compare a few actual and predicted prices
comparison = tune_df[
    ["date", "distance", "posted_rate"]
].head(10).copy()

comparison["predicted_rate"] = baseline_predictions.head(10)
comparison["absolute_error"] = (
    comparison["posted_rate"] - comparison["predicted_rate"]
).abs()

display(comparison.round(2))


from sklearn.preprocessing import OrdinalEncoder

# Build model features without changing the source data
def make_features(df, include_quote=False):
    columns = [
        "pickup", "delivery", "equipment",
        "distance", "weight",
        "pickup_lat", "pickup_lon",
        "delivery_lat", "delivery_lon",
        "market_index"
    ]

    X = df[columns].copy()
    dates = df["date"]

    # Extract calendar information
    X["weekday"] = dates.dt.dayofweek
    X["day_of_month"] = dates.dt.day
    X["day_number"] = (
        dates - pd.Timestamp("2025-01-01")
    ).dt.days

    # Represent position within the year
    X["annual_sin"] = np.sin(
        2 * np.pi * dates.dt.dayofyear / 365.25
    )
    X["annual_cos"] = np.cos(
        2 * np.pi * dates.dt.dayofyear / 365.25
    )

    X["inverse_distance"] = 1.0 / X["distance"]

    if include_quote:
        X["quote_signal"] = df["quote_signal"]

    return X


# Prepare training and model-selection inputs
X_fit = make_features(fit_df)
X_tune = make_features(tune_df)

y_fit = fit_df["posted_rate"].copy()
y_tune = tune_df["posted_rate"].copy()

# Learn category codes from training data only
categorical_columns = ["pickup", "delivery", "equipment"]

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=np.nan
)

X_fit[categorical_columns] = encoder.fit_transform(
    X_fit[categorical_columns]
)

X_tune[categorical_columns] = encoder.transform(
    X_tune[categorical_columns]
)

# Tell the model which columns represent categories
categorical_mask = [
    column in categorical_columns
    for column in X_fit.columns
]

print("Training features:", X_fit.shape)
print("Selection features:", X_tune.shape)
print("\nMissing values retained:")
print(X_fit.isna().sum()[X_fit.isna().any()])

display(X_fit.head())

from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from time import perf_counter

# Configure the first model
model = HistGradientBoostingRegressor(
    loss="absolute_error",
    learning_rate=0.06,
    max_iter=250,
    max_leaf_nodes=23,
    min_samples_leaf=45,
    l2_regularization=2.0,
    categorical_features=categorical_mask,
    early_stopping=False,
    random_state=42
)

# Train on January-June and predict July-August
start = perf_counter()

with threadpool_limits(limits=2):
    model.fit(X_fit, y_fit)
    model_predictions = model.predict(X_tune)

# Freight prices must be positive
model_predictions = np.maximum(model_predictions, 0.01)

print(f"Training and prediction time: {perf_counter() - start:.1f} seconds")

# Evaluate on the same rows used for the baseline
model_metrics = evaluate_predictions(y_tune, model_predictions)

results_table = pd.DataFrame(
    [baseline_metrics, model_metrics],
    index=["Distance baseline", "Gradient boosting"]
)

display(results_table.round(2))

mae_improvement = 100 * (
    baseline_metrics["MAE ($)"] - model_metrics["MAE ($)"]
) / baseline_metrics["MAE ($)"]

print(f"MAE reduction vs baseline: {mae_improvement:.2f}%")

from sklearn.base import clone

# Add quote_signal while preserving the existing encoded features
X_fit_quote = X_fit.copy()
X_tune_quote = X_tune.copy()

X_fit_quote["quote_signal"] = fit_df["quote_signal"]
X_tune_quote["quote_signal"] = tune_df["quote_signal"]

# The added feature is numeric, not categorical
quote_categorical_mask = categorical_mask + [False]

# Clone the configuration without copying the fitted model
model_quote = clone(model)
model_quote.set_params(
    categorical_features=quote_categorical_mask
)

with threadpool_limits(limits=2):
    model_quote.fit(X_fit_quote, y_fit)
    quote_predictions = model_quote.predict(X_tune_quote)

quote_predictions = np.maximum(quote_predictions, 0.01)

quote_metrics = evaluate_predictions(
    y_tune, quote_predictions
)

comparison_results = pd.DataFrame(
    [baseline_metrics, model_metrics, quote_metrics],
    index=[
        "Distance baseline",
        "Without quote_signal",
        "With quote_signal"
    ]
)

display(comparison_results.round(2))

# Inspect performance separately for July and August
monthly_results = []

for month in [7, 8]:
    mask = tune_df["date"].dt.month.eq(month).to_numpy()

    for name, predictions in [
        ("Without quote_signal", model_predictions),
        ("With quote_signal", quote_predictions)
    ]:
        monthly_results.append({
            "Month": month,
            "Model": name,
            **evaluate_predictions(
                y_tune.to_numpy()[mask],
                predictions[mask]
            )
        })

display(pd.DataFrame(monthly_results).round(2))


# Keep only features available for December predictions
unavailable_in_december = [
    "pickup_lat", "pickup_lon",
    "delivery_lat", "delivery_lon",
    "market_index"
]

common_feature_columns = [
    column for column in X_fit.columns
    if column not in unavailable_in_december
]

X_fit_common = X_fit[common_feature_columns].copy()
X_tune_common = X_tune[common_feature_columns].copy()

common_categorical_mask = [
    column in categorical_columns
    for column in common_feature_columns
]

# Use the same settings for a fair comparison
model_common = clone(model)
model_common.set_params(
    categorical_features=common_categorical_mask
)

with threadpool_limits(limits=2):
    model_common.fit(X_fit_common, y_fit)
    common_predictions = model_common.predict(X_tune_common)

common_predictions = np.maximum(common_predictions, 0.01)

common_metrics = evaluate_predictions(
    y_tune, common_predictions
)

selection_results = pd.DataFrame(
    [
        baseline_metrics,
        model_metrics,
        quote_metrics,
        common_metrics
    ],
    index=[
        "Distance baseline",
        "Without quote_signal",
        "With quote_signal",
        "December-compatible model"
    ]
)

print("December-compatible features:")
print(common_feature_columns)

print("\nJuly-August evaluation:")
display(selection_results.round(2))

# Combine January-August for training
pre_holdout_df = pd.concat(
    [fit_df, tune_df], ignore_index=True
)

X_pre_holdout = make_features(pre_holdout_df)
X_holdout = make_features(holdout_df)

y_pre_holdout = pre_holdout_df["posted_rate"]
y_holdout = holdout_df["posted_rate"]

# Fit a fresh encoder using January-August only
holdout_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=np.nan
)

X_pre_holdout[categorical_columns] = (
    holdout_encoder.fit_transform(
        X_pre_holdout[categorical_columns]
    )
)

X_holdout[categorical_columns] = (
    holdout_encoder.transform(
        X_holdout[categorical_columns]
    )
)

# Keep the selected model configurations unchanged
holdout_model = clone(model)
holdout_common_model = clone(model_common)

with threadpool_limits(limits=2):
    holdout_model.fit(X_pre_holdout, y_pre_holdout)
    holdout_predictions = holdout_model.predict(X_holdout)

    holdout_common_model.fit(
        X_pre_holdout[common_feature_columns],
        y_pre_holdout
    )
    holdout_common_predictions = holdout_common_model.predict(
        X_holdout[common_feature_columns]
    )

holdout_predictions = np.maximum(holdout_predictions, 0.01)
holdout_common_predictions = np.maximum(
    holdout_common_predictions, 0.01
)

# Refit the baseline using the same training period
holdout_baseline_rate = (
    pre_holdout_df["posted_rate"] / pre_holdout_df["distance"]
).median()

holdout_baseline_predictions = (
    holdout_df["distance"] * holdout_baseline_rate
)

holdout_results = pd.DataFrame(
    [
        evaluate_predictions(y_holdout, holdout_baseline_predictions),
        evaluate_predictions(y_holdout, holdout_predictions),
        evaluate_predictions(y_holdout, holdout_common_predictions)
    ],
    index=[
        "Distance baseline",
        "Selected main model",
        "December-compatible model"
    ]
)

print(f"Training rows: {len(pre_holdout_df):,}")
print(f"Holdout rows: {len(holdout_df):,}")
print("\nSeptember-October holdout results:")
display(holdout_results.round(2))

# Preserve row-level results for analysis and reporting
holdout_details = holdout_df[
    ["load_id", "date", "posted_rate"]
].copy()

holdout_details["main_prediction"] = holdout_predictions
holdout_details["common_prediction"] = holdout_common_predictions


# Prepare all labeled data and the final prediction inputs
X_all = make_features(train_clean)
y_all = train_clean["posted_rate"].copy()

X_validation = make_features(validation_clean)

# Learn category codes from all labeled data only
final_encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=np.nan
)

X_all[categorical_columns] = final_encoder.fit_transform(
    X_all[categorical_columns]
)

X_validation[categorical_columns] = final_encoder.transform(
    X_validation[categorical_columns]
)

# Refit the two locked model configurations
final_model = clone(model)
final_common_model = clone(model_common)

with threadpool_limits(limits=2):
    final_model.fit(X_all, y_all)

    validation_predictions = final_model.predict(
        X_validation[X_all.columns]
    )

    final_common_model.fit(
        X_all[common_feature_columns],
        y_all
    )

# Reject invalid numerical outputs and enforce positive prices
assert np.isfinite(validation_predictions).all()
validation_predictions = np.maximum(validation_predictions, 0.01)

# Match predictions to the template by load_id
predictions_by_id = pd.DataFrame({
    "load_id": validation_clean["load_id"].to_numpy(),
    "predicted_rate": validation_predictions
})

assert predictions_by_id["load_id"].is_unique
assert template_df["load_id"].is_unique
assert set(template_df["load_id"]) == set(predictions_by_id["load_id"])

submission = template_df[["load_id"]].merge(
    predictions_by_id,
    on="load_id",
    how="left",
    validate="one_to_one",
    sort=False
)

# Validate the required submission structure
expected_ids = {f"TE-{i:06d}" for i in range(1, 12001)}

assert len(submission) == 12000
assert set(submission["load_id"]) == expected_ids
assert list(submission.columns) == ["load_id", "predicted_rate"]
assert submission["predicted_rate"].notna().all()
assert (submission["predicted_rate"] > 0).all()

submission.to_csv(
    "validation_predictions.csv",
    index=False,
    float_format="%.2f"
)

print("Saved: validation_predictions.csv")
print(f"Predictions: {len(submission):,}")
display(submission.head())

# Build December features from the available inputs
X_december = december_clean[
    ["pickup", "delivery", "equipment", "distance", "weight"]
].copy()

dates = december_clean["date"]

X_december["weekday"] = dates.dt.dayofweek
X_december["day_of_month"] = dates.dt.day
X_december["day_number"] = (
    dates - pd.Timestamp("2025-01-01")
).dt.days

X_december["annual_sin"] = np.sin(
    2 * np.pi * dates.dt.dayofyear / 365.25
)
X_december["annual_cos"] = np.cos(
    2 * np.pi * dates.dt.dayofyear / 365.25
)
X_december["inverse_distance"] = 1.0 / X_december["distance"]

# Apply the final training encoder without refitting
X_december[categorical_columns] = final_encoder.transform(
    X_december[categorical_columns]
)

# Match the exact training feature order
X_december = X_december[common_feature_columns]

with threadpool_limits(limits=2):
    december_predictions = final_common_model.predict(X_december)

assert np.isfinite(december_predictions).all()
december_predictions = np.maximum(december_predictions, 0.01)

# Preserve the original input columns and their order
december_output = december_df.copy()
december_output["predicted_rate"] = december_predictions

expected_columns = [
    "pickup", "delivery", "distance",
    "equipment", "weight", "date", "predicted_rate"
]

assert list(december_output.columns) == expected_columns
assert len(december_output) == 31
assert (december_output["predicted_rate"] > 0).all()

# Save separately from the uploaded source file
december_output.to_csv(
    "december_predictions.csv",
    index=False,
    float_format="%.2f"
)

print("Saved: december_predictions.csv")
print(f"Predictions: {len(december_output)}")
print(f"Minimum predicted rate: ${december_predictions.min():,.2f}")
print(f"Maximum predicted rate: ${december_predictions.max():,.2f}")

display(december_output.head())

# Save evaluation evidence without changing the selected models.
from importlib.metadata import version
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
selection_results.to_csv(results_dir / "model_selection_metrics.csv", index_label="model")
holdout_results.to_csv(results_dir / "holdout_metrics.csv", index_label="model")
holdout_details.to_csv(results_dir / "holdout_predictions.csv", index=False)
pd.DataFrame(monthly_results).to_csv(results_dir / "selection_monthly.csv", index=False)
packages = ["numpy", "pandas", "matplotlib", "scikit-learn", "threadpoolctl"]
(results_dir / "environment_versions.txt").write_text(
    "\n".join(f"{name}=={version(name)}" for name in packages) + "\n", encoding="utf-8")


import subprocess
import sys
result = subprocess.run([
    sys.executable, "score.py", "--predictions", "validation_predictions.csv",
    "--december-predictions", "december_predictions.csv",
    "--output-dir", "scorer_results"
], capture_output=True, text=True)
print(result.stdout)
if result.returncode:
    print(result.stderr)
result.check_returncode()
