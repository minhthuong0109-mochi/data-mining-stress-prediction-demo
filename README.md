# Stress Prediction Demo (Django + ML)

A simple Django web demo to predict user stress risk and assign a behavior cluster profile from smartphone usage signals.  
The system also supports KNN-based micro-steps recommendations using artifacts bundled inside the calibrated risk model.

---

## Features

- **Web form UI** (`/`) to input daily usage metrics and display results
- **REST API** endpoint (`/api/predict/`) to return JSON prediction output
- **KMeans clustering** to map user into a behavior cluster + **profile name**
- **Calibrated risk model** to output:
  - stress probability
  - predicted label (0/1)
- **Recommendation (KNN micro-steps)** using:
  - `reference_pool`
  - `recommendation_config` (similarity_features, feature_weights, k_neighbors, etc.)
  stored inside the risk artifact

---

## Project Structure (high level)

- `stress_demo/` : Django project settings
- `stress_api/` : Django app (views, urls)
- `src/services/ml_service.py` : ML inference logic (load artifacts, feature engineering, predict cluster/risk, recommend micro-steps)
- `templates/index.html` : UI form + results rendering
- `resources/models/` : stored model artifacts (`*.pkl`) (optional if you keep them in root)

> Note: Make sure model artifact paths in `ml_service.py` match your folder layout.

---

## Input Fields

### Required user inputs (UI)
- `Phone_Unlocks` (times/day)
- `Awake_Time` (hours/day)
- `Sleep_Duration` (hours/day)
- `Daily_Screen_Time` (minutes/day)
- `App_Social_Media_Time` (minutes/day)
- `App_Work_Time` (minutes/day)
- `App_Entertainment_Time` (minutes/day)

### Auto-computed features (in code)
- `unlocks_per_hour`
- `screen_per_sleep`
- `unlock_intensity`
- screen ratios:
  - `screen_ratio_social = App_Social_Media_Time / Daily_Screen_Time`
  - `screen_ratio_work = App_Work_Time / Daily_Screen_Time`
  - `screen_ratio_entertainment = 1 - social - work` (or App_Entertainment_Time / Daily_Screen_Time)

---

## Outputs

The prediction returns:

- `cluster_id`
- `profile_name` (mapped from cluster_id)
- `stress_probability`
- `pred_label`
- `micro_steps` (list of recommendation actions)

