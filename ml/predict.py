"""
AI BASED FOOD SYSTEM — Prediction Module
=========================================
Loads the saved model + metadata and exposes a single
`predict_meal_demand()` function for use in Flask routes.
"""

import os
import json
import pickle
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "meal_demand_model.pkl")
META_PATH  = os.path.join(BASE_DIR, "model_metadata.json")

# ── Load once at import time (efficient for Flask) ───────────────────────────
def _load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run ml/train_model.py first."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)
    return model, metadata

_model, _metadata = _load_artifacts()


# ── Helper: encode a single categorical value ─────────────────────────────────
def _encode(column: str, value: str) -> int:
    """
    Encodes a string value using the saved LabelEncoder classes.
    Returns the integer index, or raises ValueError if unknown.
    """
    classes = _metadata["encoders"][column]["classes"]
    value = value.strip()
    if value not in classes:
        raise ValueError(
            f"Unknown {column} value: '{value}'. "
            f"Allowed values: {classes}"
        )
    return classes.index(value)


# ── Main prediction function ──────────────────────────────────────────────────
def predict_meal_demand(day_of_week: str, meal_type: str,
                        food_item: str, popularity_index: float) -> dict:
    """
    Predicts the number of students expected for a given meal.

    Parameters
    ----------
    day_of_week       : e.g. "Monday", "Tuesday" ... "Friday"
    meal_type         : e.g. "Breakfast", "Lunch", "Supper"
    food_item         : e.g. "Porridge Yam"
    popularity_index  : float between 0 and 1

    Returns
    -------
    dict with:
        predicted_students  (int)
        confidence_note     (str)
        inputs_used         (dict)
    """

    # Validate popularity index
    if not (0.0 <= float(popularity_index) <= 1.0):
        raise ValueError("popularity_index must be between 0.0 and 1.0")

    # Encode categoricals
    dow_enc  = _encode("Day_of_Week", day_of_week)
    mt_enc   = _encode("Meal_Type",   meal_type)
    fi_enc   = _encode("Food_Item",   food_item)

    # Build feature vector
    features = np.array([[dow_enc, mt_enc, fi_enc, float(popularity_index)]])

    # Predict (clamp to non-negative whole number)
    raw_pred = _model.predict(features)[0]
    predicted = max(0, int(round(raw_pred)))

    # Simple confidence note based on R²
    r2 = _metadata["metrics"]["R2"]
    if r2 >= 0.85:
        confidence = "High confidence"
    elif r2 >= 0.65:
        confidence = "Moderate confidence"
    else:
        confidence = "Low confidence — consider collecting more data"

    return {
        "predicted_students": predicted,
        "confidence_note": confidence,
        "model_r2": r2,
        "inputs_used": {
            "day_of_week":      day_of_week,
            "meal_type":        meal_type,
            "food_item":        food_item,
            "popularity_index": popularity_index
        }
    }


# ── Helper: get valid options (for frontend dropdowns) ───────────────────────
def get_valid_options() -> dict:
    """Returns allowed values for each categorical field."""
    enc = _metadata["encoders"]
    return {
        "days_of_week": enc["Day_of_Week"]["classes"],
        "meal_types":   enc["Meal_Type"]["classes"],
        "food_items":   enc["Food_Item"]["classes"],
    }
