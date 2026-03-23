"""Machine learning helpers for dsbro."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import RFE, mutual_info_classif, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    RidgeClassifier,
)
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from dsbro.metrics import classification_report, regression_report


def _validate_dataframe(df: pd.DataFrame, name: str = "df") -> pd.DataFrame:
    """Validate that an object is a DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame for {name}, got {type(df).__name__}")
    return df.copy()


def _to_frame(X: Any) -> pd.DataFrame:
    """Convert matrix-like data into a DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X.copy()
    array = np.asarray(X)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    columns = [f"x{i}" for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)


def _to_series(y: Any, name: str = "target") -> pd.Series:
    """Convert array-like targets into a Series."""
    if isinstance(y, pd.Series):
        return y.reset_index(drop=True).copy()
    return pd.Series(np.asarray(y), name=name)


def _split_features_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features and target."""
    data = _validate_dataframe(df)
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame")
    return data.drop(columns=[target]), data[target].copy()


def _infer_task(y: pd.Series, task: str = "auto") -> str:
    """Infer whether a target is classification or regression."""
    normalized = task.lower()
    if normalized in {"classification", "regression"}:
        return normalized
    if normalized != "auto":
        raise ValueError("task must be 'auto', 'classification', or 'regression'")

    if not pd.api.types.is_numeric_dtype(y):
        return "classification"
    unique_count = y.nunique(dropna=True)
    if pd.api.types.is_integer_dtype(y) and unique_count <= 20:
        return "classification"
    return "regression"


def _cv_splitter(y: pd.Series, task: str, cv: int) -> Any:
    """Create a task-aware cross-validation splitter."""
    effective_splits = min(cv, len(y))
    if task == "classification":
        min_class_count = int(y.value_counts().min())
        effective_splits = min(effective_splits, min_class_count)
        if effective_splits < 2:
            raise ValueError("Not enough samples per class for cross-validation")
        return StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=42)

    if effective_splits < 2:
        raise ValueError("Need at least two samples for cross-validation")
    return KFold(n_splits=effective_splits, shuffle=True, random_state=42)


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a preprocessing transformer for mixed-type tabular data."""
    numeric_columns = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_columns:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipe, numeric_columns))
    if categorical_columns:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipe, categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_pipeline(X: pd.DataFrame, estimator: Any) -> Pipeline:
    """Build a preprocessing + estimator pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", _make_preprocessor(X)),
            ("estimator", clone(estimator)),
        ]
    )


def _maybe_predict_probabilities(model: Any, X: pd.DataFrame, y: pd.Series) -> np.ndarray | None:
    """Return probability scores when available."""
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.ndim == 2 and probabilities.shape[1] == 2:
            return probabilities[:, 1]
        return probabilities
    if hasattr(model, "decision_function") and y.nunique(dropna=True) <= 2:
        return model.decision_function(X)
    return None


def _make_model_registry(task: str) -> dict[str, Any]:
    """Return the built-in model registry for a task."""
    if task == "classification":
        models: dict[str, Any] = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "Ridge": RidgeClassifier(),
            "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(random_state=42),
            "ExtraTrees": ExtraTreesClassifier(n_estimators=50, random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "KNN": KNeighborsClassifier(),
            "DecisionTree": DecisionTreeClassifier(random_state=42),
            "AdaBoost": AdaBoostClassifier(random_state=42, algorithm="SAMME"),
        }
    else:
        models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "RandomForest": RandomForestRegressor(n_estimators=50, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(random_state=42),
            "ExtraTrees": ExtraTreesRegressor(n_estimators=50, random_state=42),
            "SVM": SVR(),
            "KNN": KNeighborsRegressor(),
            "DecisionTree": DecisionTreeRegressor(random_state=42),
            "AdaBoost": AdaBoostRegressor(random_state=42),
        }

    for name, optional_model in _optional_models(task).items():
        models[name] = optional_model
    return models


def _optional_models(task: str) -> dict[str, Any]:
    """Load optional third-party models when installed."""
    models: dict[str, Any] = {}
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor

        models["LightGBM"] = (
            LGBMClassifier(random_state=42, verbose=-1)
            if task == "classification"
            else LGBMRegressor(random_state=42, verbose=-1)
        )
    except ImportError:
        pass

    try:
        from xgboost import XGBClassifier, XGBRegressor

        models["XGBoost"] = (
            XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss")
            if task == "classification"
            else XGBRegressor(random_state=42, verbosity=0)
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier, CatBoostRegressor

        models["CatBoost"] = (
            CatBoostClassifier(random_state=42, verbose=0)
            if task == "classification"
            else CatBoostRegressor(random_state=42, verbose=0)
        )
    except ImportError:
        pass

    return models


def _normalize_model_name(name: str) -> str:
    """Normalize a model name for lookups."""
    return "".join(character for character in name.lower() if character.isalnum())


def _resolve_estimator(model: Any, task: str) -> tuple[str, Any]:
    """Resolve a model name or estimator into a concrete estimator."""
    registry = _make_model_registry(task)
    if isinstance(model, str):
        normalized = _normalize_model_name(model)
        aliases = {
            _normalize_model_name(key): key for key in registry
        }
        aliases.update(
            {
                "rf": "RandomForest",
                "randomforestclassifier": "RandomForest",
                "randomforestregressor": "RandomForest",
                "logistic": "LogisticRegression",
                "linear": "LinearRegression",
                "dt": "DecisionTree",
                "lgbm": "LightGBM",
                "lightgbm": "LightGBM",
                "xgb": "XGBoost",
                "xgboost": "XGBoost",
            }
        )
        if normalized not in aliases:
            available = ", ".join(sorted(registry))
            raise ValueError(f"Unknown model '{model}'. Available models: {available}")
        resolved_name = aliases[normalized]
        if resolved_name not in registry:
            raise ImportError(
                f"{resolved_name} required. Install with: pip install dsbro[ml]"
            )
        return resolved_name, registry[resolved_name]

    return model.__class__.__name__, clone(model)


def _summarize_folds(fold_scores: pd.DataFrame) -> dict[str, float]:
    """Summarize fold-level metrics into mean and std values."""
    summary: dict[str, float] = {}
    numeric_columns = [column for column in fold_scores.columns if column != "fold"]
    for column in numeric_columns:
        summary[f"{column}_mean"] = float(fold_scores[column].mean())
        summary[f"{column}_std"] = float(fold_scores[column].std(ddof=0))
    return summary


def _primary_metric(task: str) -> str:
    """Return the primary optimization metric for a task."""
    return "accuracy" if task == "classification" else "rmse"


def _primary_sort_ascending(task: str) -> bool:
    """Return whether the primary metric should be sorted ascending."""
    return task == "regression"


def compare(
    df: pd.DataFrame,
    target: str,
    task: str = "auto",
    cv: int = 5,
) -> pd.DataFrame:
    """Compare multiple models with cross-validation.

    Args:
        df: Input DataFrame containing features and target.
        target: Target column name.
        task: ``auto``, ``classification``, or ``regression``.
        cv: Number of cross-validation folds.

    Returns:
        A leaderboard DataFrame sorted by the primary metric.

    Example:
        >>> import pandas as pd
        >>> from dsbro.ml import compare
        >>> frame = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
        >>> leaderboard = compare(frame, target="y", cv=2)
        >>> "model" in leaderboard.columns
        True
    """
    X, y = _split_features_target(df, target)
    inferred_task = _infer_task(y, task)
    rows: list[dict[str, Any]] = []

    for model_name, estimator in _make_model_registry(inferred_task).items():
        try:
            result = cross_validate(estimator, X, y, cv=cv, metric="auto")
            rows.append(
                {
                    "model": model_name,
                    "task": inferred_task,
                    "status": "ok",
                    **result["summary"],
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "model": model_name,
                    "task": inferred_task,
                    "status": "error",
                    "error": str(exc),
                }
            )

    leaderboard = pd.DataFrame(rows)
    primary = f"{_primary_metric(inferred_task)}_mean"
    if primary in leaderboard.columns:
        leaderboard = leaderboard.sort_values(
            by=["status", primary],
            ascending=[True, _primary_sort_ascending(inferred_task)],
            na_position="last",
        )
    return leaderboard.reset_index(drop=True)


def cross_validate(
    model: Any,
    X: Any,
    y: Any,
    cv: int = 5,
    metric: str = "auto",
) -> dict[str, Any]:
    """Run manual cross-validation for an estimator.

    Args:
        model: Estimator instance or model name.
        X: Feature matrix.
        y: Target values.
        cv: Number of folds.
        metric: Primary metric selector. ``auto`` uses task defaults.

    Returns:
        A dictionary containing fold scores and summary metrics.

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from dsbro.ml import cross_validate
        >>> X = [[0], [1], [2], [3]]
        >>> y = [0, 0, 1, 1]
        >>> result = cross_validate(LogisticRegression(max_iter=500), X, y, cv=2)
        >>> "summary" in result
        True
    """
    X_frame = _to_frame(X)
    y_series = _to_series(y)
    inferred_task = _infer_task(y_series, "auto")
    _, estimator = _resolve_estimator(model, inferred_task)
    splitter = _cv_splitter(y_series, inferred_task, cv)

    fold_rows: list[dict[str, Any]] = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X_frame, y_series), start=1):
        X_train = X_frame.iloc[train_idx]
        X_valid = X_frame.iloc[valid_idx]
        y_train = y_series.iloc[train_idx]
        y_valid = y_series.iloc[valid_idx]

        pipeline = _build_pipeline(X_train, estimator)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_valid)

        if inferred_task == "classification":
            probabilities = _maybe_predict_probabilities(pipeline, X_valid, y_series)
            metrics = classification_report(y_valid, predictions, probabilities)
        else:
            metrics = regression_report(y_valid, predictions, n_features=X_frame.shape[1])
        metrics["fold"] = fold
        fold_rows.append(metrics)

    fold_scores = pd.DataFrame(fold_rows)
    summary = _summarize_folds(fold_scores)
    primary = _primary_metric(inferred_task) if metric == "auto" else metric
    summary["primary_metric"] = primary
    summary["task"] = inferred_task
    return {"fold_scores": fold_scores, "summary": summary}


def oof_predict(model: Any, X: Any, y: Any, cv: int = 5) -> np.ndarray:
    """Generate out-of-fold predictions.

    Args:
        model: Estimator instance or model name.
        X: Feature matrix.
        y: Target values.
        cv: Number of folds.

    Returns:
        A numpy array of out-of-fold predictions with the same length as ``y``.

    Example:
        >>> from sklearn.linear_model import LogisticRegression
        >>> from dsbro.ml import oof_predict
        >>> X = [[0], [1], [2], [3]]
        >>> y = [0, 0, 1, 1]
        >>> preds = oof_predict(LogisticRegression(max_iter=500), X, y, cv=2)
        >>> len(preds)
        4
    """
    X_frame = _to_frame(X)
    y_series = _to_series(y)
    inferred_task = _infer_task(y_series, "auto")
    _, estimator = _resolve_estimator(model, inferred_task)
    splitter = _cv_splitter(y_series, inferred_task, cv)

    if inferred_task == "regression":
        oof = np.zeros(len(y_series), dtype=float)
    elif y_series.nunique(dropna=True) <= 2:
        oof = np.zeros(len(y_series), dtype=float)
    else:
        oof = np.empty(len(y_series), dtype=object)

    for train_idx, valid_idx in splitter.split(X_frame, y_series):
        X_train = X_frame.iloc[train_idx]
        X_valid = X_frame.iloc[valid_idx]
        y_train = y_series.iloc[train_idx]
        pipeline = _build_pipeline(X_train, estimator)
        pipeline.fit(X_train, y_train)

        if inferred_task == "classification" and y_series.nunique(dropna=True) <= 2:
            probabilities = _maybe_predict_probabilities(pipeline, X_valid, y_series)
            if probabilities is not None and np.asarray(probabilities).ndim == 1:
                oof[valid_idx] = probabilities
            else:
                oof[valid_idx] = pipeline.predict(X_valid)
        else:
            oof[valid_idx] = pipeline.predict(X_valid)

    return oof


def train(
    df: pd.DataFrame,
    target: str,
    model: Any = "lgbm",
    params: dict[str, Any] | None = None,
    cv: int = 5,
) -> tuple[Pipeline, np.ndarray, dict[str, Any]]:
    """Train a single model with cross-validation diagnostics.

    Args:
        df: Input DataFrame containing features and target.
        target: Target column name.
        model: Estimator instance or model name.
        params: Optional estimator parameters applied before fitting.
        cv: Number of folds used for diagnostics.

    Returns:
        A tuple of ``(fitted_pipeline, oof_predictions, metrics_summary)``.

    Example:
        >>> import pandas as pd
        >>> from dsbro.ml import train
        >>> frame = pd.DataFrame({"x": [1, 2, 3, 4], "y": [0, 0, 1, 1]})
        >>> fitted, preds, metrics = train(frame, target="y", model="logistic", cv=2)
        >>> len(preds)
        4
    """
    X, y = _split_features_target(df, target)
    inferred_task = _infer_task(y, "auto")
    model_name, estimator = _resolve_estimator(model, inferred_task)
    if params:
        estimator.set_params(**params)

    cv_result = cross_validate(estimator, X, y, cv=cv)
    oof_predictions = oof_predict(estimator, X, y, cv=cv)
    final_model = _build_pipeline(X, estimator)
    final_model.fit(X, y)
    summary = {"model": model_name, **cv_result["summary"]}
    return final_model, oof_predictions, summary


def _estimator_grid_key(estimator: Any) -> str:
    """Return a normalized estimator key for tuning grids."""
    return estimator.__class__.__name__.lower()


def _default_param_grid(estimator: Any) -> dict[str, list[Any]]:
    """Return a small default parameter grid for a supported estimator."""
    key = _estimator_grid_key(estimator)
    grids = {
        "logisticregression": {"estimator__C": [0.1, 1.0, 10.0]},
        "linearregression": {"estimator__fit_intercept": [True, False]},
        "ridge": {"estimator__alpha": [0.1, 1.0, 10.0]},
        "ridgeclassifier": {"estimator__alpha": [0.1, 1.0, 10.0]},
        "randomforestclassifier": {
            "estimator__n_estimators": [50, 100],
            "estimator__max_depth": [None, 5],
        },
        "randomforestregressor": {
            "estimator__n_estimators": [50, 100],
            "estimator__max_depth": [None, 5],
        },
        "gradientboostingclassifier": {
            "estimator__n_estimators": [50, 100],
            "estimator__learning_rate": [0.05, 0.1],
        },
        "gradientboostingregressor": {
            "estimator__n_estimators": [50, 100],
            "estimator__learning_rate": [0.05, 0.1],
        },
        "extratreesclassifier": {
            "estimator__n_estimators": [50, 100],
            "estimator__max_depth": [None, 5],
        },
        "extratreesregressor": {
            "estimator__n_estimators": [50, 100],
            "estimator__max_depth": [None, 5],
        },
        "svc": {"estimator__C": [0.1, 1.0, 10.0]},
        "svr": {"estimator__C": [0.1, 1.0, 10.0]},
        "kneighborsclassifier": {"estimator__n_neighbors": [3, 5, 7]},
        "kneighborsregressor": {"estimator__n_neighbors": [3, 5, 7]},
        "decisiontreeclassifier": {"estimator__max_depth": [None, 3, 5]},
        "decisiontreeregressor": {"estimator__max_depth": [None, 3, 5]},
        "adaboostclassifier": {
            "estimator__n_estimators": [25, 50],
            "estimator__learning_rate": [0.5, 1.0],
        },
        "adaboostregressor": {
            "estimator__n_estimators": [25, 50],
            "estimator__learning_rate": [0.5, 1.0],
        },
    }
    if key not in grids:
        raise ValueError(f"No default tuning grid available for {estimator.__class__.__name__}")
    return grids[key]


def tune(
    model: Any,
    X: Any,
    y: Any,
    method: str = "optuna",
    n_trials: int = 100,
) -> dict[str, Any]:
    """Tune an estimator with Optuna or GridSearchCV.

    Args:
        model: Estimator instance or model name.
        X: Feature matrix.
        y: Target values.
        method: ``optuna`` or ``gridsearch``.
        n_trials: Trial count for Optuna. Ignored by GridSearchCV.

    Returns:
        A dictionary with the best fitted model, parameters, and score.

    Example:
        >>> from dsbro.ml import tune
        >>> result = tune("logistic", [[0], [1], [2], [3]], [0, 0, 1, 1], method="gridsearch")
        >>> "best_params" in result
        True
    """
    X_frame = _to_frame(X)
    y_series = _to_series(y)
    inferred_task = _infer_task(y_series, "auto")
    model_name, estimator = _resolve_estimator(model, inferred_task)
    normalized_method = method.lower()

    if normalized_method == "optuna":
        try:
            import optuna
        except ImportError:
            normalized_method = "gridsearch"
        else:
            param_grid = _default_param_grid(estimator)
            splitter = _cv_splitter(y_series, inferred_task, min(5, len(y_series)))
            scoring = (
                "accuracy"
                if inferred_task == "classification"
                else "neg_root_mean_squared_error"
            )

            def objective(trial: Any) -> float:
                trial_params = {}
                for param_name, values in param_grid.items():
                    if all(isinstance(value, (int, np.integer)) for value in values):
                        trial_params[param_name] = trial.suggest_int(
                            param_name,
                            int(min(values)),
                            int(max(values)),
                        )
                    else:
                        trial_params[param_name] = trial.suggest_categorical(param_name, values)
                pipeline = _build_pipeline(X_frame, estimator)
                pipeline.set_params(**trial_params)
                search = GridSearchCV(
                    pipeline,
                    param_grid={key: [value] for key, value in trial_params.items()},
                    cv=splitter,
                    scoring=scoring,
                )
                search.fit(X_frame, y_series)
                return float(search.best_score_)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=min(n_trials, 20))
            best_params = study.best_params
            best_pipeline = _build_pipeline(X_frame, estimator)
            best_pipeline.set_params(**best_params)
            best_pipeline.fit(X_frame, y_series)
            return {
                "best_model": best_pipeline,
                "best_params": best_params,
                "best_score": float(study.best_value),
                "method": "optuna",
                "model": model_name,
            }

    if normalized_method != "gridsearch":
        raise ValueError("method must be 'optuna' or 'gridsearch'")

    param_grid = _default_param_grid(estimator)
    scoring = "accuracy" if inferred_task == "classification" else "neg_root_mean_squared_error"
    splitter = _cv_splitter(y_series, inferred_task, min(5, len(y_series)))
    pipeline = _build_pipeline(X_frame, estimator)
    search = GridSearchCV(pipeline, param_grid=param_grid, cv=splitter, scoring=scoring)
    search.fit(X_frame, y_series)
    return {
        "best_model": search.best_estimator_,
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "method": "gridsearch",
        "model": model_name,
    }


def blend(
    predictions: list[Any] | tuple[Any, ...],
    weights: list[float] | tuple[float, ...] | None = None,
    method: str = "weighted",
) -> np.ndarray:
    """Blend multiple prediction arrays.

    Args:
        predictions: Sequence of prediction arrays.
        weights: Optional weights used by weighted blending.
        method: ``weighted``, ``rank``, or ``power``.

    Returns:
        A blended numpy array.

    Example:
        >>> from dsbro.ml import blend
        >>> blend([[0.1, 0.2], [0.3, 0.4]]).shape[0]
        2
    """
    stacked = np.vstack([np.asarray(prediction, dtype=float) for prediction in predictions])
    if stacked.ndim != 2:
        raise ValueError("predictions must be 1D arrays of equal length")

    normalized_method = method.lower()
    if weights is None:
        weights_array = np.ones(stacked.shape[0], dtype=float) / stacked.shape[0]
    else:
        weights_array = np.asarray(weights, dtype=float)
        weights_array = weights_array / weights_array.sum()

    if normalized_method == "weighted":
        return np.average(stacked, axis=0, weights=weights_array)
    if normalized_method == "rank":
        ranks = np.vstack([pd.Series(row).rank(method="average").to_numpy() for row in stacked])
        return np.average(ranks, axis=0, weights=weights_array)
    if normalized_method == "power":
        return power_mean(stacked, p=1.5)
    raise ValueError("method must be 'weighted', 'rank', or 'power'")


def power_mean(predictions: Any, p: float = 1.0) -> np.ndarray:
    """Compute the generalized mean across prediction arrays."""
    stacked = np.asarray(predictions, dtype=float)
    if stacked.ndim == 1:
        return stacked
    if p == 0:
        clipped = np.clip(stacked, 1e-12, None)
        return np.exp(np.mean(np.log(clipped), axis=0))
    return np.power(np.mean(np.power(stacked, p), axis=0), 1.0 / p)


def stack(
    base_preds: list[Any] | tuple[Any, ...],
    meta_model: Any,
    y: Any,
) -> tuple[Any, np.ndarray]:
    """Fit a meta-model on base predictions for stacking."""
    meta_features = np.column_stack([np.asarray(pred) for pred in base_preds])
    y_series = _to_series(y)
    task = _infer_task(y_series, "auto")
    _, estimator = _resolve_estimator(meta_model, task)
    fitted = clone(estimator)
    fitted.fit(meta_features, y_series)

    if task == "classification" and hasattr(fitted, "predict_proba") and y_series.nunique() <= 2:
        stacked_predictions = fitted.predict_proba(meta_features)[:, 1]
    else:
        stacked_predictions = fitted.predict(meta_features)
    return fitted, np.asarray(stacked_predictions)


def feature_select(
    X: Any,
    y: Any,
    method: str = "importance",
    top_n: int = 20,
) -> pd.DataFrame:
    """Score features using a simple selection strategy.

    Args:
        X: Feature matrix.
        y: Target values.
        method: ``importance``, ``mutual_info``, ``boruta``, or ``rfe``.
        top_n: Maximum number of top-ranked features to flag as selected.

    Returns:
        A feature-importance DataFrame sorted by score.

    Example:
        >>> from dsbro.ml import feature_select
        >>> scores = feature_select([[0], [1], [2], [3]], [0, 0, 1, 1])
        >>> "feature" in scores.columns
        True
    """
    X_frame = _to_frame(X)
    y_series = _to_series(y)
    task = _infer_task(y_series, "auto")
    numeric = X_frame.select_dtypes(include=["number", "bool"]).copy()
    if numeric.empty:
        raise ValueError("feature_select currently requires numeric feature columns")

    normalized_method = method.lower()
    if normalized_method == "importance":
        estimator = (
            RandomForestClassifier(n_estimators=100, random_state=42)
            if task == "classification"
            else RandomForestRegressor(n_estimators=100, random_state=42)
        )
        estimator.fit(numeric.fillna(0), y_series)
        scores = estimator.feature_importances_
    elif normalized_method == "mutual_info":
        if task == "classification":
            scores = mutual_info_classif(numeric.fillna(0), y_series, random_state=42)
        else:
            scores = mutual_info_regression(numeric.fillna(0), y_series, random_state=42)
    elif normalized_method == "rfe":
        estimator = (
            LogisticRegression(max_iter=1000, random_state=42)
            if task == "classification"
            else LinearRegression()
        )
        selector = RFE(estimator, n_features_to_select=min(top_n, numeric.shape[1]))
        selector.fit(numeric.fillna(0), y_series)
        scores = 1 / selector.ranking_
    elif normalized_method == "boruta":
        try:
            from boruta import BorutaPy
        except ImportError as exc:
            raise ImportError("Boruta required. Install with: pip install boruta") from exc
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = BorutaPy(estimator, n_estimators="auto", random_state=42)
        selector.fit(numeric.fillna(0).to_numpy(), y_series.to_numpy())
        scores = selector.ranking_
        scores = 1 / np.asarray(scores)
    else:
        raise ValueError("method must be 'importance', 'mutual_info', 'boruta', or 'rfe'")

    summary = pd.DataFrame({"feature": numeric.columns, "score": scores})
    summary = summary.sort_values("score", ascending=False).reset_index(drop=True)
    summary["selected"] = summary.index < min(top_n, len(summary))
    return summary


def pseudo_label(
    model: Any,
    X_train: Any,
    y_train: Any,
    X_test: Any,
    threshold: float = 0.95,
) -> dict[str, Any]:
    """Augment labeled data with confident pseudo-labels from test predictions."""
    X_train_frame = _to_frame(X_train)
    y_train_series = _to_series(y_train)
    X_test_frame = _to_frame(X_test)
    task = _infer_task(y_train_series, "auto")
    _, estimator = _resolve_estimator(model, task)
    pipeline = _build_pipeline(X_train_frame, estimator)
    pipeline.fit(X_train_frame, y_train_series)

    if task == "classification" and hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test_frame)
        confidence = probabilities.max(axis=1)
        pseudo_labels = pipeline.classes_[np.argmax(probabilities, axis=1)]
        selected_mask = confidence >= threshold
    else:
        pseudo_labels = pipeline.predict(X_test_frame)
        selected_mask = np.ones(len(X_test_frame), dtype=bool)

    pseudo_X = X_test_frame.loc[selected_mask].copy()
    pseudo_y = pd.Series(np.asarray(pseudo_labels)[selected_mask], name=y_train_series.name)
    X_augmented = pd.concat([X_train_frame, pseudo_X], ignore_index=True)
    y_augmented = pd.concat([y_train_series.reset_index(drop=True), pseudo_y], ignore_index=True)
    return {
        "model": pipeline,
        "X_augmented": X_augmented,
        "y_augmented": y_augmented,
        "pseudo_count": int(selected_mask.sum()),
    }


def adversarial_validation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cv: int = 5,
) -> dict[str, Any]:
    """Measure train-vs-test separability with a classifier."""
    train_frame = _validate_dataframe(train, "train")
    test_frame = _validate_dataframe(test, "test")
    combined = pd.concat([train_frame, test_frame], ignore_index=True, sort=False)
    labels = pd.Series([0] * len(train_frame) + [1] * len(test_frame), name="dataset")
    result = cross_validate("logistic", combined, labels, cv=cv)
    summary = result["summary"]
    return {
        "auc_mean": float(summary.get("auc_mean", np.nan)),
        "fold_scores": result["fold_scores"],
        "summary": summary,
    }


def auto_train(
    df: pd.DataFrame,
    target: str,
    time_budget: int = 300,
) -> dict[str, Any]:
    """Run a simple automatic train-and-rank workflow."""
    del time_budget
    leaderboard = compare(df, target=target, task="auto", cv=3)
    successful = leaderboard[leaderboard["status"] == "ok"]
    if successful.empty:
        raise RuntimeError("No models completed successfully during compare()")

    best_model_name = successful.iloc[0]["model"]
    model, oof_predictions, metrics = train(df, target=target, model=best_model_name, cv=3)
    task = metrics["task"]
    return {
        "task": task,
        "best_model_name": best_model_name,
        "leaderboard": leaderboard,
        "model": model,
        "oof_predictions": oof_predictions,
        "metrics": metrics,
    }


__all__ = [
    "adversarial_validation",
    "auto_train",
    "blend",
    "compare",
    "cross_validate",
    "feature_select",
    "oof_predict",
    "power_mean",
    "pseudo_label",
    "stack",
    "train",
    "tune",
]
