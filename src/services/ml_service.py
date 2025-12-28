# src/services/ml_service.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

# Django settings integration
try:
    from django.conf import settings

    BASE_DIR = Path(settings.BASE_DIR)
except Exception:
    # Fallback for standalone execution
    BASE_DIR = Path(__file__).resolve().parents[2]


# CONFIGURATION
MODELS_DIR = BASE_DIR / "resources" / "models"
KMEANS_PATH = MODELS_DIR / "kmeans_model.pkl"
RISK_PATH = MODELS_DIR / "risk_score_model_calibrated.pkl"

# Cluster profile mapping
CLUSTER_PROFILES = {
    0: "Nhóm nghiện lướt mạng xã hội (Social Scrollers)",
    1: "Nhóm nghiện công việc (Work Mode)",
    2: "Nhóm người dùng tối giản (Light Users)",
    3: "Nhóm người dùng đa nhiệm, luôn online (Always On)",
    4: "Nhóm xem liên mạch và ngủ bù (Binge & Sleep)"
}

_EPS = 1e-6


# DATA CLASSES
@dataclass
class LoadedModels:
    kmeans_bundle: Dict[str, Any]
    risk_bundle: Dict[str, Any]


_models_cache: Optional[LoadedModels] = None

# MODEL LOADING
def load_models() -> LoadedModels:
    """
    Load ML models once and cache.
    Thread-safe for Django multi-process environments.
    """
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    kmeans_bundle = joblib.load(KMEANS_PATH)
    risk_bundle = joblib.load(RISK_PATH)

    _models_cache = LoadedModels(
        kmeans_bundle=kmeans_bundle,
        risk_bundle=risk_bundle
    )
    return _models_cache


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _to_float(x: Any, key: str) -> float:
    # None
    if x is None:
        return 0.0

    # Already numeric
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    # String
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return 0.0
        try:
            return float(s)
        except ValueError:
            raise ValueError(f"Cannot convert '{key}' value '{s}' to float")

    # List/tuple: take first element if single item
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return 0.0
        if len(x) == 1:
            return _to_float(x[0], key)
        raise TypeError(
            f"Feature '{key}' got list/tuple with {len(x)} items: {x}. "
            f"Expected single value."
        )

    # Dict: try common value keys
    if isinstance(x, dict):
        for cand in ("value", "v", "val", "data"):
            if cand in x:
                return _to_float(x[cand], key)
        raise TypeError(
            f"Feature '{key}' is a dict but missing standard keys "
            f"('value', 'v', 'val'). Got: {x}"
        )

    # Unknown type
    raise TypeError(
        f"Feature '{key}' must be numeric, got {type(x).__name__}: {x}"
    )


def _safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    """Safe division with epsilon"""
    return float(a) / (float(b) + eps)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# FEATURE ENGINEERING
def _calc_ratios(payload: Dict[str, float]) -> Tuple[float, float, float]:
    daily = _to_float(payload.get("Daily_Screen_Time", 0), "Daily_Screen_Time")

    if daily <= 0:
        # Return zeros if no screen time
        return 0.0, 0.0, 0.0

    social = _to_float(payload.get("App_Social_Media_Time", 0), "App_Social_Media_Time")
    work = _to_float(payload.get("App_Work_Time", 0), "App_Work_Time")
    ent = _to_float(payload.get("App_Entertainment_Time", 0), "App_Entertainment_Time")

    screen_ratio_social = _safe_div(social, daily, _EPS)
    screen_ratio_work = _safe_div(work, daily, _EPS)
    screen_ratio_ent = _safe_div(ent, daily, _EPS)

    return screen_ratio_social, screen_ratio_work, screen_ratio_ent


def _build_feature_value(payload: Dict[str, float], feat: str) -> float:
    # Direct feature (already in payload)
    if feat in payload:
        return _to_float(payload[feat], feat)

    # Extract base features
    phone_unlocks = _to_float(payload.get("Phone_Unlocks", 0), "Phone_Unlocks")
    sleep_duration = _to_float(payload.get("Sleep_Duration", 0), "Sleep_Duration")
    daily_screen_time = _to_float(payload.get("Daily_Screen_Time", 0), "Daily_Screen_Time")
    awake_time = _to_float(payload.get("Awake_Time", 24 - sleep_duration), "Awake_Time")

    app_social = _to_float(payload.get("App_Social_Media_Time", 0), "App_Social_Media_Time")
    app_work = _to_float(payload.get("App_Work_Time", 0), "App_Work_Time")
    app_ent = _to_float(payload.get("App_Entertainment_Time", 0), "App_Entertainment_Time")

    # Engineered features
    if feat == "unlocks_per_hour":
        return _safe_div(phone_unlocks, awake_time, _EPS)

    if feat == "screen_per_sleep":
        return _safe_div(daily_screen_time, sleep_duration, 0.1)

    if feat == "unlock_intensity":
        return _safe_div(phone_unlocks, awake_time, 0.1)

    if feat == "screen_ratio_social":
        return _safe_div(app_social, daily_screen_time, _EPS)

    if feat == "screen_ratio_work":
        return _safe_div(app_work, daily_screen_time, _EPS)

    if feat == "screen_ratio_ent":
        return _safe_div(app_ent, daily_screen_time, _EPS)

    # Unknown feature
    raise KeyError(f"Unknown feature '{feat}' in similarity vector")


# PREDICTION FUNCTIONS

# Mapping cluster -> profile name
CLUSTER_PROFILES = {
    0: "Nhóm nghiện lướt mạng xã hội (Social Scrollers)",
    1: "Nhóm nghiện công việc (Work-Driven High Screen)",
    2: "Nhóm người dùng tối giản (Mindful Digital Balance)",
    3: "Nhóm người dùng đa nhiệm, luôn online (Restless Overload)",
}

_EPS = 1e-6


@dataclass
class LoadedModels:
    kmeans_bundle: Dict[str, Any]
    risk_bundle: Dict[str, Any]


_models_cache: LoadedModels | None = None


def load_models() -> LoadedModels:
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    kmeans_bundle = joblib.load(KMEANS_PATH)
    risk_bundle = joblib.load(RISK_PATH)

    _models_cache = LoadedModels(kmeans_bundle=kmeans_bundle, risk_bundle=risk_bundle)
    return _models_cache


def _calc_ratios(payload: Dict[str, float]) -> Tuple[float, float, float]:
    daily = float(payload["Daily_Screen_Time"])
    if daily <= 0:
        raise ValueError("Daily_Screen_Time must be > 0")

    social = float(payload["App_Social_Media_Time"])
    work = float(payload["App_Work_Time"])
    ent = float(payload["App_Entertainment_Time"])

    screen_ratio_social = social / (daily + _EPS)
    screen_ratio_work = work / (daily + _EPS)
    screen_ratio_ent = ent / (daily + _EPS)
    return screen_ratio_social, screen_ratio_work, screen_ratio_ent


def _predict_cluster(payload: Dict[str, float], kmeans_bundle: Dict[str, Any]) -> int:
    screen_ratio_social, screen_ratio_work, _ = _calc_ratios(payload)

    X = pd.DataFrame([{
        "Sleep_Duration": float(payload["Sleep_Duration"]),
        "Daily_Screen_Time": float(payload["Daily_Screen_Time"]),
        "Phone_Unlocks": float(payload["Phone_Unlocks"]),
        "screen_ratio_social": screen_ratio_social,
        "screen_ratio_work": screen_ratio_work,
    }])

    scaler = kmeans_bundle.get("scaler")
    model = kmeans_bundle.get("model")

    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    # model có thể là KMeans hoặc Pipeline
    cluster_id = int(model.predict(X_scaled)[0])
    return cluster_id


def _predict_risk(payload: Dict[str, float], cluster_id: int, risk_bundle: Dict[str, Any]) -> Tuple[float, int]:
    phone_unlocks = float(payload["Phone_Unlocks"])
    awake_time = float(payload["Awake_Time"])
    sleep_duration = float(payload["Sleep_Duration"])
    daily_screen_time = float(payload["Daily_Screen_Time"])

    unlocks_per_hour = phone_unlocks / (awake_time + _EPS)
    screen_per_sleep = daily_screen_time / (sleep_duration + 0.1)
    unlock_intensity = phone_unlocks / (awake_time + 0.1)

    # model calibrated thường nằm ở key 'calibrated_model'
    estimator = risk_bundle.get("calibrated_model") or risk_bundle.get("model")
    if estimator is None:
        raise ValueError("Risk model not found in risk bundle (expected 'calibrated_model' or 'model').")

    row = pd.DataFrame([{
        "Phone_Unlocks": phone_unlocks,
        "unlocks_per_hour": unlocks_per_hour,
        "Awake_Time": awake_time,
        "screen_per_sleep": screen_per_sleep,
        "unlock_intensity": unlock_intensity,
        "Sleep_Duration": sleep_duration,
        "cluster_id": float(cluster_id),
    }])

    proba = float(estimator.predict_proba(row)[0][1])

    threshold = float(risk_bundle.get("optimal_threshold", 0.5))
    pred_label = int(proba >= threshold)
    return proba, pred_label


# RECOMMENDATION SYSTEM
def recommend_micro_steps(
        feature_payload: Dict[str, float],
        risk_artifact: Dict[str, Any],
        user_cluster_id: Optional[int] = None,
        user_risk_score: Optional[float] = None,
        top_n: int = 3
) -> List[str]:

    # Extract recommendation components from artifact
    cfg = risk_artifact.get("recommendation_config", {})
    ref = risk_artifact.get("reference_pool", {})

    if not cfg or not ref:
        return []

    # Get configuration
    similarity_features: List[str] = cfg.get("similarity_features", [])

    feature_weights_dict: Dict[str, float] = cfg.get("feature_weights", {})

    # Convert dict to list (ordered by similarity_features)
    feature_weights_list = [
        feature_weights_dict.get(feat, 1.0)
        for feat in similarity_features
    ]

    sim_config = cfg.get("similarity_config", {})
    k_neighbors: int = int(sim_config.get("k_neighbors", 5))

    if not similarity_features:
        return []

    # Load reference pool
    pool_raw = np.array(ref.get("vectors_raw", []), dtype=float)
    pool_scaled = np.array(ref.get("vectors_scaled", []), dtype=float)
    pool_risk_scores = np.array(ref.get("risk_scores", []), dtype=float)

    if pool_raw.size == 0 or pool_scaled.size == 0:
        return []

    # Build user behavior vector
    try:
        user_raw = np.array([
            _build_feature_value(feature_payload, f)
            for f in similarity_features
        ], dtype=float)
    except Exception as e:
        print(f"Error building user vector: {e}")
        return []

    # Scale user vector (using pool statistics)
    mean = pool_raw.mean(axis=0)
    std = pool_raw.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    user_scaled = (user_raw - mean) / std

    # Filter candidates by risk score
    if user_risk_score is not None and pool_risk_scores.size > 0:
        # Only consider neighbors with lower stress
        mask = pool_risk_scores < user_risk_score

        # Fallback: relax filter if insufficient neighbors
        if mask.sum() < 3:
            relax_threshold = cfg.get("fallback_config", {}).get("relax_threshold", 0.2)
            mask = pool_risk_scores < (user_risk_score + relax_threshold)

        candidate_indices = np.where(mask)[0]
    else:
        candidate_indices = np.arange(pool_scaled.shape[0])

    if len(candidate_indices) == 0:
        return []

    # Compute weighted cosine similarity
    W = np.array(feature_weights_list, dtype=float)
    W = W / (W.sum() + 1e-9)  # Normalize weights
    w_sqrt = np.sqrt(W)

    user_vec = user_scaled * w_sqrt

    similarities = []
    for idx in candidate_indices:
        neighbor_vec = pool_scaled[idx] * w_sqrt
        sim = _cosine_similarity(user_vec, neighbor_vec)
        similarities.append((idx, sim))

    # Get top-k neighbors
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_indices = [idx for idx, _ in similarities[:k_neighbors]]

    if len(top_indices) == 0:
        return []

    # Aggregate neighbor profile (mean in RAW space)
    neighbor_raw = pool_raw[top_indices]
    target_profile = neighbor_raw.mean(axis=0)

    # Generate micro-steps
    feature_thresholds = cfg.get("feature_thresholds", {})
    step_caps = cfg.get("step_caps", {})
    templates = cfg.get("templates", {})

    suggestions = []

    for j, feat in enumerate(similarity_features):
        current_value = float(user_raw[j])
        target_value = float(target_profile[j])
        diff = target_value - current_value  # + means increase, - means decrease

        # Check if difference exceeds threshold
        threshold = float(feature_thresholds.get(feat, 0.0))
        if abs(diff) < threshold:
            continue

        # Generate micro-step based on feature type

        # Sleep_Duration: suggest increase if diff > 0
        if feat == "Sleep_Duration" and diff > 0:
            caps = step_caps.get(feat, {})
            min_step = float(caps.get("min", 0.3))
            max_step = float(caps.get("max", 1.5))
            step = max(min_step, min(max_step, diff))
            unit = caps.get("unit", "giờ")

            tpl = templates.get(feat, {}).get("increase", "")
            if tpl:
                text = tpl.format(step=step, unit=unit, target=target_value)
                suggestions.append((abs(diff), text))
            continue

        # Daily_Screen_Time: suggest decrease if diff < 0
        if feat == "Daily_Screen_Time" and diff < 0:
            caps = step_caps.get(feat, {})
            pct_min = float(caps.get("pct_min", 0.10))
            pct_max = float(caps.get("pct_max", 0.25))
            unit = caps.get("unit", "phút")

            base = max(current_value, 1.0)
            step_pct = abs(diff) / base
            step_pct = max(pct_min, min(pct_max, step_pct))
            step = base * step_pct

            tpl = templates.get(feat, {}).get("decrease", "")
            if tpl:
                text = tpl.format(step=step, unit=unit, diff=abs(diff), target=target_value)
                suggestions.append((abs(diff), text))
            continue

        # Phone_Unlocks: suggest decrease if diff < 0
        if feat == "Phone_Unlocks" and diff < 0:
            caps = step_caps.get(feat, {})
            pct_min = float(caps.get("pct_min", 0.10))
            pct_max = float(caps.get("pct_max", 0.20))
            unit = caps.get("unit", "lần")

            base = max(current_value, 1.0)
            step_pct = abs(diff) / base
            step_pct = max(pct_min, min(pct_max, step_pct))
            step = base * step_pct
            target_unlocks = max(0.0, current_value - step)

            tpl = templates.get(feat, {}).get("decrease", "")
            if tpl:
                text = tpl.format(target=target_unlocks, unit=unit, diff=abs(diff))
                suggestions.append((abs(diff), text))
            continue

        # Screen ratios: suggest decrease if diff < 0
        if feat.startswith("screen_ratio") and diff < 0:
            caps = step_caps.get(feat, {})
            pct_min = float(caps.get("pct_min", 0.05))
            pct_max = float(caps.get("pct_max", 0.15))
            unit = caps.get("unit", "%")

            diff_pct = abs(diff) * 100.0
            step = max(pct_min * 100.0, min(pct_max * 100.0, diff_pct))

            tpl = templates.get(feat, {}).get("decrease", "")
            if tpl:
                text = tpl.format(step=step, unit=unit)
                suggestions.append((abs(diff), text))
            continue

    # Sort by importance (larger diff first) and take top N
    suggestions.sort(key=lambda x: x[0], reverse=True)
    micro_steps = [text for _, text in suggestions[:top_n]]

    return micro_steps


# MAIN PREDICTION API

def predict(payload: Dict[str, float]) -> Dict[str, Any]:
    # Load models
    models = load_models()

    # Step 1: Predict cluster
    cluster_id = _predict_cluster(payload, models.kmeans_bundle)
    profile_name = CLUSTER_PROFILES.get(cluster_id, f"Profile {cluster_id}")

    # Step 2: Predict risk
    stress_proba, pred_label = _predict_risk(payload, cluster_id, models.risk_bundle)

    # Step 3: Generate recommendations (only if risk > threshold)
    micro_steps = []
    if stress_proba > 0.30:  # Only recommend for risk > 30%
        micro_steps = recommend_micro_steps(
            feature_payload=payload,
            risk_artifact=models.risk_bundle,
            user_cluster_id=cluster_id,
            user_risk_score=stress_proba,
            top_n=3
        )

    return {
        "cluster_id": cluster_id,
        "profile_name": profile_name,
        "stress_probability": round(stress_proba, 4),
        "pred_label": pred_label,
        "micro_steps": micro_steps,
    }
