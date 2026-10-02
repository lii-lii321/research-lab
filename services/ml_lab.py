"""ML 基线实验：特征守门、自动任务推断、多模型真实训练与指标对比。"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = XGBRegressor = None  # type: ignore[misc,assignment]
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = LGBMRegressor = None  # type: ignore[misc,assignment]

from models.schemas import ExcludedFeature, MLExperimentResult, MLModelResult
from services.tracking import TrackingStore, dataframe_fingerprint

SEED = 42
TEST_SIZE = 0.2
MIN_ROWS = 30
HIGH_MISSING = 0.5
MAX_CARDINALITY = 50
K_MIN, K_MAX = 2, 6
SILHOUETTE_SAMPLE_CAP = 5000
CLASSIFICATION_MAX_CLASSES = 20

TASKS = ("auto", "regression", "classification", "clustering")


class DataProblem(Exception):
    """数据本身无法支撑实验（样本不足等），应转化为结构化失败结果。"""


def _regression_models():
    models = [
        ("LinearRegression", LinearRegression(), {}),
        (
            "RandomForest",
            RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1),
            {"n_estimators": 100},
        ),
    ]
    if XGBRegressor is not None:
        models.append(
            (
                "XGBoost",
                XGBRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.1,
                    random_state=SEED, n_jobs=-1, verbosity=0,
                ),
                {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1},
            )
        )
    if LGBMRegressor is not None:
        models.append(
            (
                "LightGBM",
                LGBMRegressor(n_estimators=200, random_state=SEED, n_jobs=-1, verbose=-1),
                {"n_estimators": 200},
            )
        )
    return models


def _classification_models():
    models = [
        ("LogisticRegression", LogisticRegression(max_iter=500), {"max_iter": 500}),
        (
            "RandomForest",
            RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
            {"n_estimators": 100},
        ),
    ]
    if XGBClassifier is not None:
        models.append(
            (
                "XGBoost",
                XGBClassifier(
                    n_estimators=200, max_depth=4, learning_rate=0.1,
                    random_state=SEED, n_jobs=-1, verbosity=0,
                ),
                {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.1},
            )
        )
    if LGBMClassifier is not None:
        models.append(
            (
                "LightGBM",
                LGBMClassifier(n_estimators=200, random_state=SEED, n_jobs=-1, verbose=-1),
                {"n_estimators": 200},
            )
        )
    return models


def select_features(df: pd.DataFrame, profile, target: str | None):
    numeric: list[str] = []
    categorical: list[str] = []
    excluded: list[ExcludedFeature] = []
    for c in profile.columns:
        name = c.name
        if target is not None and name == target:
            continue
        if c.type == "identifier":
            excluded.append(ExcludedFeature(column=name, reason="标识符列"))
        elif c.missing_rate > HIGH_MISSING:
            excluded.append(ExcludedFeature(column=name, reason=f"缺失率 {c.missing_rate:.0%}"))
        elif c.type == "numeric":
            numeric.append(name)
        elif c.type in ("categorical", "boolean"):
            if c.n_unique <= MAX_CARDINALITY:
                categorical.append(name)
            else:
                excluded.append(ExcludedFeature(column=name, reason=f"唯一值过多（{c.n_unique}）"))
        elif c.type == "datetime":
            excluded.append(ExcludedFeature(column=name, reason="时间列暂不参与"))
        else:
            excluded.append(ExcludedFeature(column=name, reason="文本或空列"))
    return numeric, categorical, excluded


def infer_task(profile, target: str) -> str:
    col = next((c for c in profile.columns if c.name == target), None)
    if col is None:
        raise ValueError(f"目标列 {target} 不在数据集中")
    if col.type == "numeric":
        if col.n_unique <= CLASSIFICATION_MAX_CLASSES and col.unique_ratio < 0.05:
            return "classification"
        return "regression"
    if col.type in ("categorical", "boolean"):
        return "classification"
    raise ValueError(f"目标列 {target} 类型为 {col.type}，不能作为监督学习目标")


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers)


def _supervised_metrics(
    task: str, y_test: pd.Series, pred: np.ndarray, proba: np.ndarray | None, n_classes: int
) -> dict[str, float]:
    if task == "regression":
        return {
            "MAE": round(float(mean_absolute_error(y_test, pred)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(y_test, pred))), 4),
            "R2": round(float(r2_score(y_test, pred)), 4),
        }
    metrics = {
        "Accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "Precision_macro": round(float(precision_score(y_test, pred, average="macro", zero_division=0)), 4),
        "Recall_macro": round(float(recall_score(y_test, pred, average="macro", zero_division=0)), 4),
        "F1_macro": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 4),
    }
    if n_classes == 2 and proba is not None:
        metrics["ROC_AUC"] = round(float(roc_auc_score(y_test, proba)), 4)
    return metrics


def _fold_primary(task: str, y_true: pd.Series, pred: np.ndarray) -> float:
    if task == "regression":
        return float(r2_score(y_true, pred))
    return float(f1_score(y_true, pred, average="macro", zero_division=0))


def _cv_scores(
    task: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric: list[str],
    categorical: list[str],
    estimator,
    k: int,
) -> tuple[str, list[float]]:
    """分层 k 折（分类按 y 分层）交叉验证，返回主指标名与每折得分。"""
    primary = "R2" if task == "regression" else "F1_macro"
    if task == "classification":
        splitter = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
    else:
        splitter = KFold(n_splits=k, shuffle=True, random_state=SEED)
    base = Pipeline([("prep", _preprocessor(numeric, categorical)), ("model", estimator)])
    scores: list[float] = []
    for tr, va in splitter.split(X_train, y_train):
        pipe = clone(base)
        pipe.fit(X_train.iloc[tr], y_train.iloc[tr])
        pred = pipe.predict(X_train.iloc[va])
        scores.append(_fold_primary(task, y_train.iloc[va], pred))
    return primary, scores


def _run_supervised(df: pd.DataFrame, task: str, target: str, numeric: list[str], categorical: list[str]):
    feature_cols = numeric + categorical
    if target not in df.columns:
        raise ValueError(f"目标列 {target} 不在数据集中")
    if task == "regression":
        y_full = pd.to_numeric(df[target], errors="coerce")
    else:
        y_full = df[target].astype(str).where(df[target].notna(), other=None)
    mask = y_full.notna()
    X_df = df.loc[mask, feature_cols]
    y = y_full[mask]
    if len(y) < MIN_ROWS:
        raise DataProblem(f"有效样本不足（{len(y)} < {MIN_ROWS}）")
    stratify_arg = None
    y_model = y
    n_classes = 0
    if task == "classification":
        counts = y.value_counts()
        if counts.size < 2:
            raise DataProblem(f"目标列只有 {counts.size} 个类别，无法分类")
        if counts.min() >= 2:
            stratify_arg = y
        encoder = LabelEncoder()
        y_model = pd.Series(encoder.fit_transform(y), index=y.index)
        n_classes = int(counts.size)
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_model, test_size=TEST_SIZE, random_state=SEED, stratify=stratify_arg
    )
    cv_k = 5 if len(X_train) >= 100 else 3
    if task == "classification":
        cv_k = min(cv_k, int(y_train.value_counts().min()))
    use_cv = cv_k >= 2
    preprocessor = _preprocessor(numeric, categorical)
    results: list[MLModelResult] = []
    for name, estimator, params in (
        _regression_models() if task == "regression" else _classification_models()
    ):
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        started = time.perf_counter()
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1] if n_classes == 2 else None
        seconds = time.perf_counter() - started
        metrics = _supervised_metrics(task, y_test, pred, proba, n_classes)
        if use_cv:
            primary, scores = _cv_scores(
                task, X_train, y_train, numeric, categorical, estimator, cv_k
            )
            metrics[f"{primary}_CV"] = round(float(np.mean(scores)), 4)
            metrics[f"{primary}_CVsd"] = (
                round(float(np.std(scores, ddof=1)), 4) if len(scores) > 1 else 0.0
            )
        results.append(
            MLModelResult(
                model=name,
                params=params,
                metrics=metrics,
                train_seconds=round(seconds, 3),
            )
        )
    return results, int(len(X_train)), int(len(X_test))


def _run_clustering(df: pd.DataFrame, numeric: list[str]):
    X = df[numeric].apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if len(X) < MIN_ROWS:
        raise DataProblem(f"完整数值样本不足（{len(X)} < {MIN_ROWS}）")
    X_scaled = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    ).fit_transform(X)
    scored = []
    for k in range(K_MIN, min(K_MAX, len(X_scaled))):
        model = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(
            X_scaled, labels,
            sample_size=min(len(X_scaled), SILHOUETTE_SAMPLE_CAP), random_state=SEED,
        )
        scored.append((float(score), model, labels))
    best_score, best_model, best_labels = max(scored, key=lambda t: t[0])
    sizes = pd.Series(best_labels).value_counts().sort_index()
    result = MLModelResult(
        model=f"KMeans(k={best_model.n_clusters})",
        params={"n_clusters": int(best_model.n_clusters), "random_state": SEED},
        metrics={"silhouette": round(best_score, 4), "inertia": round(float(best_model.inertia_), 2)},
    )
    sizes_dict = {f"簇 {int(i)}": int(c) for i, c in sizes.items()}
    return [result], int(best_model.n_clusters), sizes_dict


def _failed(task: str, reason: str) -> MLExperimentResult:
    return MLExperimentResult(status="failed", reason=reason, task=task)


def run_ml_experiment(
    df: pd.DataFrame,
    profile,
    target: str | None = None,
    task: str = "auto",
    store: TrackingStore | None = None,
    dataset_name: str = "dataset",
) -> MLExperimentResult:
    started = time.perf_counter()
    if task not in TASKS:
        raise ValueError(f"未知任务类型：{task}（可选 {' / '.join(TASKS)}）")
    if task == "auto":
        task = infer_task(profile, target) if target else "clustering"
    if task == "clustering":
        target = None
        numeric, categorical, excluded = select_features(df, profile, None)
        if len(numeric) < 2:
            return _failed(task, f"聚类至少需要 2 个数值特征（当前 {len(numeric)} 个）")
        try:
            models, k, sizes = _run_clustering(df, numeric)
        except DataProblem as exc:
            return _failed(task, str(exc))
        n_train = n_test = 0
        best_metric_name = "silhouette"
    else:
        if not target:
            raise ValueError("回归 / 分类任务必须提供目标列")
        numeric, categorical, excluded = select_features(df, profile, target)
        if not numeric + categorical:
            return _failed(task, "没有可用特征（标识符 / 高缺失 / 文本列被排除）")
        try:
            models, n_train, n_test = _run_supervised(df, task, target, numeric, categorical)
        except DataProblem as exc:
            return _failed(task, str(exc))
        k = None
        sizes = {}
        best_metric_name = "R2" if task == "regression" else "F1_macro"
    # 单次 holdout 排名近乎抽签：有 CV 时按 CV 均值选最佳模型
    cv_key = f"{best_metric_name}_CV"
    key_metric = cv_key if any(cv_key in m.metrics for m in models) else best_metric_name
    best = max(models, key=lambda m: m.metrics.get(key_metric, float("-inf")))
    result = MLExperimentResult(
        status="ok",
        task=task,
        target=target,
        features_numeric=numeric,
        features_categorical=categorical,
        excluded=excluded,
        n_train=n_train,
        n_test=n_test,
        n_clusters=k,
        cluster_sizes=sizes,
        models=models,
        best_model=best.model,
        best_metric_name=best_metric_name,
        best_metric_value=best.metrics.get(best_metric_name),
        dataset_fingerprint=dataframe_fingerprint(df),
        runtime_seconds=round(time.perf_counter() - started, 3),
    )
    if store is not None:
        notes = "；".join(f"{e.column}: {e.reason}" for e in excluded)[:500]
        result.tracked_uid = store.track(
            kind="ml",
            task=task,
            target=target,
            dataset_name=dataset_name,
            df=df,
            feature_set={
                "numeric": numeric,
                "categorical": categorical,
                "n_clusters": k,
            },
            models_results=[m.model_dump() for m in models],
            best_model=result.best_model,
            best_metric_name=best_metric_name,
            best_metric_value=result.best_metric_value,
            notes=notes,
            runtime_seconds=result.runtime_seconds,
        )
    return result
