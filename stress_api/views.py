# stress_api/views.py
"""
Django Views for Stress Prediction System
WITH DATABASE LOGGING - saves all predictions to SQLite
"""

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.db import connection

from src.services.ml_service import predict

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

TABLE_NAME = 'stress_predictions'


def ensure_table_exists():
    """
    Create the stress_predictions table if it does not exist.

    Schema:
    - 7 input features (REAL)
    - 3 prediction outputs (REAL/TEXT)
    - 1 timestamp (DATETIME)
    """
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone_unlocks REAL NOT NULL,
                awake_time REAL NOT NULL,
                sleep_duration REAL NOT NULL,
                daily_screen_time REAL NOT NULL,
                app_social_media_time REAL NOT NULL,
                app_work_time REAL NOT NULL,
                app_entertainment_time REAL NOT NULL,
                cluster_id INTEGER NOT NULL,
                profile_name TEXT NOT NULL,
                stress_probability REAL NOT NULL,
                pred_label INTEGER NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


def save_prediction(input_features: dict, prediction_result: dict):
    """
    Persist a single prediction sample to SQLite.

    Args:
        input_features: Dict with 7 input features
        prediction_result: Dict with prediction outputs (cluster_id, profile_name, stress_probability, pred_label)
    """
    ensure_table_exists()

    sql = (
        f"INSERT INTO {TABLE_NAME} ("
        f"phone_unlocks, awake_time, sleep_duration, daily_screen_time, "
        f"app_social_media_time, app_work_time, app_entertainment_time, "
        f"cluster_id, profile_name, stress_probability, pred_label"
        f") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    )

    params = [
        float(input_features["Phone_Unlocks"]),
        float(input_features["Awake_Time"]),
        float(input_features["Sleep_Duration"]),
        float(input_features["Daily_Screen_Time"]),
        float(input_features["App_Social_Media_Time"]),
        float(input_features["App_Work_Time"]),
        float(input_features["App_Entertainment_Time"]),
        int(prediction_result["cluster_id"]),
        str(prediction_result["profile_name"]),
        float(prediction_result["stress_probability"]),
        int(prediction_result["pred_label"])
    ]

    with connection.cursor() as cursor:
        cursor.execute(sql, params)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _parse_payload(request: HttpRequest) -> dict:
    """Extract form data from POST or GET request"""
    src = request.POST if request.method == "POST" else request.GET
    return {k: src.get(k) for k in src.keys()}


def _to_float(payload: dict, keys: list[str]) -> dict:
    """Convert specified keys to float, raise error if missing"""
    out = dict(payload)
    for k in keys:
        if k not in out or out[k] in (None, ""):
            raise ValueError(f"Missing field: {k}")
        out[k] = float(out[k])
    return out


# ============================================================================
# WEB INTERFACE VIEW
# ============================================================================

def demo_page(request: HttpRequest) -> HttpResponse:
    """
    Main web interface for stress prediction.
    Handles form submission and displays results.
    """
    context = {}

    if request.method == "POST":
        try:
            # Parse and validate input
            payload = _parse_payload(request)
            payload = _to_float(payload, [
                "Phone_Unlocks", "Awake_Time", "Sleep_Duration", "Daily_Screen_Time",
                "App_Social_Media_Time", "App_Work_Time", "App_Entertainment_Time",
            ])

            # Predict
            result = predict(payload)

            # Convert stress_probability to percentage (0.3285 -> 32.85)
            result["stress_probability"] = result["stress_probability"] * 100

            # Save to database (BEFORE converting percentage - save original 0-1 scale)
            saved = True
            try:
                # Create a copy with original probability for DB
                result_for_db = dict(result)
                result_for_db["stress_probability"] = result["stress_probability"] / 100  # Convert back to 0-1
                save_prediction(payload, result_for_db)
            except Exception as e:
                # Do not fail the request if logging fails
                saved = False
                print(f"WARNING: Failed to save prediction to database: {e}")

            context["result"] = result
            context["saved_to_db"] = saved  # Optional: show in UI

            # Save submitted data to refill form
            context["submitted_data"] = payload

            # Round integers for display
            for key in ["Phone_Unlocks", "Daily_Screen_Time",
                        "App_Social_Media_Time", "App_Work_Time", "App_Entertainment_Time"]:
                if key in context["submitted_data"]:
                    context["submitted_data"][key] = int(float(context["submitted_data"][key]))

        except Exception as e:
            context["error"] = str(e)

            # Save data even on error (for form refill)
            try:
                context["submitted_data"] = _parse_payload(request)
            except:
                pass

    return render(request, "index.html", context)


# ============================================================================
# JSON API ENDPOINT
# ============================================================================

@csrf_exempt
def predict_api(request: HttpRequest) -> JsonResponse:
    """
    JSON API endpoint for stress prediction.

    POST /api/predict/
    Body: {
        "Phone_Unlocks": 120,
        "Awake_Time": 18.0,
        "Sleep_Duration": 6.0,
        "Daily_Screen_Time": 420,
        "App_Social_Media_Time": 180,
        "App_Work_Time": 150,
        "App_Entertainment_Time": 90
    }

    Response: {
        "cluster_id": 1,
        "profile_name": "...",
        "stress_probability": 0.3285,
        "pred_label": 0,
        "micro_steps": [...],
        "saved": true
    }
    """
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)

    try:
        # Parse input
        payload = request.POST.dict()
        payload = _to_float(payload, [
            "Phone_Unlocks", "Awake_Time", "Sleep_Duration", "Daily_Screen_Time",
            "App_Social_Media_Time", "App_Work_Time", "App_Entertainment_Time",
        ])

        # Predict
        result = predict(payload)

        # Save to database
        saved = True
        try:
            save_prediction(payload, result)
        except Exception as e:
            # Do not fail the request if logging fails
            saved = False
            print(f"WARNING: Failed to save prediction to database: {e}")

        # Add saved status to response
        result["saved"] = saved

        return JsonResponse(result, status=200)

    except ValueError as e:
        return JsonResponse({"error": str(e)}, status=400)
    except Exception as e:
        return JsonResponse({"error": f"Prediction failed: {str(e)}"}, status=500)


# ============================================================================
# OPTIONAL: DATABASE STATS VIEW
# ============================================================================

def stats_view(request: HttpRequest) -> JsonResponse:
    """
    Optional endpoint to view database statistics.
    GET /api/stats/
    """
    try:
        ensure_table_exists()

        with connection.cursor() as cursor:
            # Total predictions
            cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
            total_predictions = cursor.fetchone()[0]

            # Average stress probability
            cursor.execute(f"SELECT AVG(stress_probability) FROM {TABLE_NAME}")
            avg_stress = cursor.fetchone()[0] or 0.0

            # Cluster distribution
            cursor.execute(
                f"SELECT cluster_id, profile_name, COUNT(*) as count "
                f"FROM {TABLE_NAME} GROUP BY cluster_id, profile_name"
            )
            cluster_stats = [
                {"cluster_id": row[0], "profile_name": row[1], "count": row[2]}
                for row in cursor.fetchall()
            ]

            # High risk count (stress_probability > 0.6)
            cursor.execute(
                f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE stress_probability > 0.6"
            )
            high_risk_count = cursor.fetchone()[0]

        return JsonResponse({
            "total_predictions": total_predictions,
            "average_stress_probability": round(avg_stress, 4),
            "high_risk_count": high_risk_count,
            "cluster_distribution": cluster_stats
        })

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)