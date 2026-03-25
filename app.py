import os
import sys
import pandas as pd

root_path = os.path.abspath(os.path.dirname(__file__))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import json
import subprocess
from flask import Flask, request, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# ── Database config ───────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')

# Use DATABASE_URL from Render/env, fallback to local default
default_db_uri = 'postgresql://postgres:AIbased@localhost:5432/babcock_db'
db_url = os.getenv('DATABASE_URL', default_db_uri)

# Render uses 'postgres://' but SQLAlchemy requires 'postgresql://'
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = db_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db     = SQLAlchemy(app)
bcrypt = Bcrypt(app)

CORS(app, origins="*")


app.config.update(
    SESSION_COOKIE_SAMESITE='None',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=True,
)

# ── ML model loader ───────────────────────────────────────────────────────────
try:
    from ml.predict import predict_meal_demand, get_valid_options
    ML_AVAILABLE = True
    print("✅ ML model loaded successfully.")
except (ImportError, ModuleNotFoundError, FileNotFoundError) as e:
    ML_AVAILABLE = False
    print(f"⚠️  ML model not available: {e}")

# ── Models ────────────────────────────────────────────────────────────────────
class Admin(db.Model):
    __tablename__ = 'administrator'
    id         = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50),  nullable=False)
    last_name  = db.Column(db.String(50),  nullable=False)
    email      = db.Column(db.String(120), unique=True, nullable=False)
    department = db.Column(db.String(100), nullable=False)
    password   = db.Column(db.String(255), nullable=False)
    avatar     = db.Column(db.Text,        nullable=True)
    otp        = db.Column(db.String(10),  nullable=True)

    def to_dict(self):
        return {
            "id": self.id, "first_name": self.first_name,
            "last_name": self.last_name, "email": self.email,
            "department": self.department, "avatar": self.avatar
        }


class MealRecord(db.Model):
    """
    Stores each prediction + actual students served.
    New rows are appended to meal_dataset.xlsx and used to retrain the model.
    """
    __tablename__ = 'meal_records'

    id                 = db.Column(db.Integer,     primary_key=True)
    record_date        = db.Column(db.Date,        nullable=True)
    day_of_week        = db.Column(db.String(20),  nullable=False)
    meal_type          = db.Column(db.String(20),  nullable=False)
    food_item          = db.Column(db.String(100), nullable=False)
    popularity_index   = db.Column(db.Float,       nullable=True)
    predicted_students = db.Column(db.Integer,     nullable=True)
    actual_students    = db.Column(db.Integer,     nullable=False)

    def to_dict(self):
        return {
            "id":                 self.id,
            "record_date":        str(self.record_date) if self.record_date else None,
            "day_of_week":        self.day_of_week,
            "meal_type":          self.meal_type,
            "food_item":          self.food_item,
            "popularity_index":   self.popularity_index,
            "predicted_students": self.predicted_students,
            "actual_students":    self.actual_students,
        }


with app.app_context():
    db.create_all()


# ══════════════════════════════════════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "ml_model_loaded": ML_AVAILABLE}), 200

@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.json
    if Admin.query.filter_by(email=data.get('email')).first():
        return jsonify({"error": "Email already registered"}), 400
    hashed = bcrypt.generate_password_hash(data.get('password')).decode('utf-8')
    new_admin = Admin(
        first_name=data.get('firstName'), last_name=data.get('lastName'),
        email=data.get('email'), department=data.get('department'), password=hashed
    )
    try:
        db.session.add(new_admin); db.session.commit()
        return jsonify({"message": "Admin registered successfully"}), 201
    except Exception as e:
        db.session.rollback(); return jsonify({"error": str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    data  = request.json
    admin = Admin.query.filter_by(email=data.get('email')).first()
    if admin and bcrypt.check_password_hash(admin.password, data.get('password')):
        session['user_id']    = admin.id
        session['user_email'] = admin.email
        return jsonify({"message": "Login successful", "user": admin.to_dict()}), 200
    return jsonify({"error": "Invalid email or password"}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"message": "Logged out successfully"}), 200

@app.route('/api/profile', methods=['GET'])
def get_profile():
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    admin = db.session.get(Admin, session['user_id'])
    if not admin: return jsonify({"error": "Admin not found"}), 404
    return jsonify(admin.to_dict()), 200

@app.route('/api/profile/update', methods=['POST'])
def update_profile():
    if 'user_id' not in session: return jsonify({"error": "Unauthorized"}), 401
    data  = request.json
    admin = db.session.get(Admin, session['user_id'])
    if not admin: return jsonify({"error": "Admin not found"}), 404
    admin.first_name = data.get('firstName', admin.first_name)
    admin.last_name  = data.get('lastName',  admin.last_name)
    admin.department = data.get('department', admin.department)
    new_email = data.get('email', admin.email)
    if new_email != admin.email:
        if Admin.query.filter_by(email=new_email).first():
            return jsonify({"error": "Email already in use"}), 400
        admin.email = new_email
    try:
        db.session.commit()
        return jsonify({"message": "Profile updated", "user": admin.to_dict()}), 200
    except Exception as e:
        db.session.rollback(); return jsonify({"error": str(e)}), 500

@app.route('/api/profile/password', methods=['POST'])
def update_password():
    if 'user_id' not in session: return jsonify({"error": "Unauthorized"}), 401
    data  = request.json
    admin = db.session.get(Admin, session['user_id'])
    if not bcrypt.check_password_hash(admin.password, data.get('currentPassword')):
        return jsonify({"error": "Incorrect current password"}), 400
    admin.password = bcrypt.generate_password_hash(data.get('newPassword')).decode('utf-8')
    try:
        db.session.commit(); return jsonify({"message": "Password updated"}), 200
    except Exception as e:
        db.session.rollback(); return jsonify({"error": str(e)}), 500

@app.route('/api/profile/upload-avatar', methods=['POST'])
def upload_avatar():
    if 'user_id' not in session: return jsonify({"error": "Unauthorized"}), 401
    data  = request.json
    admin = db.session.get(Admin, session['user_id'])
    admin.avatar = data.get('avatar')
    try:
        db.session.commit()
        return jsonify({"message": "Avatar updated", "avatar": admin.avatar}), 200
    except Exception as e:
        db.session.rollback(); return jsonify({"error": str(e)}), 500

@app.route('/api/dashboard', methods=['GET'])
def dashboard_data():
    if 'user_id' not in session: return jsonify({"error": "Unauthorized"}), 401
    return jsonify({"stats": {"total_meals": 5842, "expected_students": 6120, "risk_level": "Low"}}), 200


# ══════════════════════════════════════════════════════════════════════════════
# ML ROUTES
# ══════════════════════════════════════════════════════════════════════════════

@app.route('/api/predict', methods=['POST'])
def predict():
    if not ML_AVAILABLE:
        return jsonify({"error": "ML model not loaded. Run ml/train_model.py first."}), 503
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json or {}
    required = ["day_of_week", "meal_type", "food_item"]
    missing  = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # ── Auto-calculate popularity index ──────────────────────────────────────
    # Use the average actual_students for this food item from DB records,
    # normalised against the overall max, as a proxy for popularity.
    # Falls back to the Excel dataset average if no DB records exist yet.
    popularity_index = _calculate_popularity(data["food_item"])

    try:
        result = predict_meal_demand(
            day_of_week=data["day_of_week"],
            meal_type=data["meal_type"],
            food_item=data["food_item"],
            popularity_index=popularity_index
        )
        result["popularity_index_used"] = popularity_index  # send back to frontend
        return jsonify(result), 200
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


def _calculate_popularity(food_item: str) -> float:
    """
    Calculates a popularity index (0.0–1.0) for a food item.

    Priority order:
    1. Average actual_students for this food item in DB, normalised by
       the max average across all food items  →  reflects real-world data
    2. Falls back to the value in the Excel seed dataset if no DB rows exist
    3. Returns 0.5 (neutral) if neither source has data
    """
    import pandas as pd

    # 1. Try DB records first
    from sqlalchemy import func
    rows = (
        db.session.query(
            MealRecord.food_item,
            func.avg(MealRecord.actual_students).label("avg_students")
        )
        .filter(MealRecord.actual_students.isnot(None))
        .group_by(MealRecord.food_item)
        .all()
    )

    if rows:
        averages   = {r.food_item: float(r.avg_students) for r in rows}
        max_avg    = max(averages.values()) or 1
        popularity = averages.get(food_item)
        if popularity is not None:
            return round(popularity / max_avg, 4)

    # 2. Fall back to Excel seed data
    excel_path = os.path.join(os.path.dirname(__file__), "ml", "meal_dataset.xlsx")
    if os.path.exists(excel_path):
        try:
            df = pd.read_excel(excel_path, engine="openpyxl")
            df.columns = df.columns.str.strip()
            if "Food_Item" in df.columns and "Popularity_Index" in df.columns:
                match = df[df["Food_Item"].str.strip() == food_item.strip()]
                if not match.empty:
                    return round(float(match["Popularity_Index"].mean()), 4)
        except Exception:
            pass

    # 3. Neutral fallback
    return 0.5


@app.route('/api/predict/options', methods=['GET'])
def predict_options():
    if not ML_AVAILABLE:
        return jsonify({"error": "ML model not loaded."}), 503
    return jsonify(get_valid_options()), 200


@app.route('/api/actuals', methods=['POST'])
def save_actuals():
    """
    POST /api/actuals
    -----------------
    1. Saves actual post-meal data to PostgreSQL (meal_records table)
    2. Exports the combined DB + original dataset back to ml/meal_dataset.xlsx
    3. Retrains the model with the updated data
    4. Hot-reloads the model so the next /api/predict uses the improved version
    """
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.json or {}
    required = ["day_of_week", "meal_type", "food_item", "actual_students"]
    missing  = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Parse optional date
    record_date = None
    if data.get("date"):
        try:
            from datetime import datetime
            record_date = datetime.strptime(data["date"], "%Y-%m-%d").date()
        except ValueError:
            pass

    # 1. Save to PostgreSQL
    record = MealRecord(
        record_date        = record_date,
        day_of_week        = data["day_of_week"],
        meal_type          = data["meal_type"],
        food_item          = data["food_item"],
        popularity_index   = data.get("popularity_index"),
        predicted_students = data.get("predicted_students"),
        actual_students    = int(data["actual_students"]),
    )
    try:
        db.session.add(record)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"DB save failed: {str(e)}"}), 500

    # 2. Export dataset to Excel
    try:
        _export_dataset_to_excel()
    except Exception as e:
        return jsonify({
            "message": "Actuals saved but dataset export failed.",
            "warning": str(e)
        }), 207

    # 3. Retrain model
    new_metrics = _retrain_model()

    return jsonify({
        "message": "Actuals saved, dataset updated, model retrained.",
        "record":      record.to_dict(),
        "new_metrics": new_metrics
    }), 200


def _export_dataset_to_excel():
    """
    Saves actual records from DB into ml/actuals_log.xlsx.

    Strategy:
      - ml/meal_dataset.xlsx  → NEVER touched (original seed, read-only)
      - ml/actuals_log.xlsx   → appended every time a user submits actuals
      - train_model.py reads BOTH files and combines them at training time
    """
    import pandas as pd
    from sqlalchemy import func

    BASE_DIR    = os.path.join(os.path.dirname(__file__), "ml")
    ACTUALS_LOG = os.path.join(BASE_DIR, "actuals_log.xlsx")

    # ── Pull ALL actual records from DB ──────────────────────────────────────
    records = MealRecord.query.filter(
        MealRecord.actual_students.isnot(None)
    ).all()

    if not records:
        print("📊 No actual records in DB yet")
        return

    # ── Auto-calculate popularity index from DB averages ─────────────────────
    avg_rows = (
        db.session.query(
            MealRecord.food_item,
            func.avg(MealRecord.actual_students).label("avg_students")
        )
        .filter(MealRecord.actual_students.isnot(None))
        .group_by(MealRecord.food_item)
        .all()
    )
    avg_map = {r.food_item: float(r.avg_students) for r in avg_rows}
    max_avg = max(avg_map.values()) if avg_map else 1

    def get_popularity(record):
        if record.popularity_index is not None:
            return float(record.popularity_index)
        avg = avg_map.get(record.food_item, max_avg * 0.5)
        return round(min(1.0, avg / max_avg), 4)

    # ── Write actuals_log.xlsx — one row per DB record ────────────────────────
    df_actuals = pd.DataFrame([{
        "Day_of_Week":       r.day_of_week,
        "Meal_Type":         r.meal_type,
        "Food_Item":         r.food_item,
        "Popularity_Index":  get_popularity(r),
        "Expected_Students": r.actual_students,
        "Date":              str(r.record_date) if r.record_date else "",
        "Predicted":         r.predicted_students or "",
    } for r in records])

    df_actuals.to_excel(ACTUALS_LOG, index=False, engine="openpyxl")
    print(f"📋 Actuals log saved: {len(df_actuals)} rows → {ACTUALS_LOG}")


def _retrain_model():
    """
    Runs ml/train_model.py as a subprocess, then hot-reloads
    the predict module so the running Flask app uses the new model.
    Returns new metrics dict or None on failure.
    """
    import importlib
    train_script = os.path.join(os.path.dirname(__file__), "ml", "train_model.py")

    try:
        # Force UTF-8 encoding so emojis and arrows don't crash Windows
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            ["python", train_script],
            capture_output=True, text=True, encoding="utf-8", env=env, timeout=120
        )
        if result.returncode != 0:
            print(f"⚠️  Retrain error:\n{result.stderr}")
            return None

        # Hot-reload predict module
        import ml.predict as pred_module
        importlib.reload(pred_module)

        global predict_meal_demand, get_valid_options, ML_AVAILABLE
        predict_meal_demand = pred_module.predict_meal_demand
        get_valid_options   = pred_module.get_valid_options
        ML_AVAILABLE        = True

        meta_path = os.path.join(os.path.dirname(__file__), "ml", "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                return json.load(f).get("metrics")

    except subprocess.TimeoutExpired:
        print("⚠️  Retrain timed out")
    except Exception as e:
        print(f"⚠️  Retrain failed: {e}")

    return None



@app.route('/api/actuals/records', methods=['GET'])
def get_actuals_records():
    """
    Returns all stored actual records from PostgreSQL.
    Also confirms whether actuals_log.xlsx exists and how many rows it has.
    GET /api/actuals/records
    """
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    # Fetch all records from DB
    records = MealRecord.query.order_by(MealRecord.id.desc()).all()

    # Check actuals_log.xlsx
    import pandas as pd
    actuals_log_path = os.path.join(os.path.dirname(__file__), "ml", "actuals_log.xlsx")
    excel_info = {"exists": False, "rows": 0}

    if os.path.exists(actuals_log_path):
        try:
            df = pd.read_excel(actuals_log_path, engine="openpyxl")
            excel_info = {"exists": True, "rows": len(df)}
        except Exception as e:
            excel_info = {"exists": True, "rows": 0, "error": str(e)}

    return jsonify({
        "total_records": len(records),
        "actuals_log":   excel_info,
        "records":       [r.to_dict() for r in records]
    }), 200


@app.route('/api/dashboard/chart', methods=['GET'])
def dashboard_chart():
    """
    Returns total predicted students per day for the past 7 days.
    Used to power the bar chart on the Dashboard.
    Reads from MealRecord DB (real predictions made by the system).
    Falls back to the dataset averages if no DB records exist yet.
    """
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    import pandas as pd
    from datetime import date, timedelta

    today = date.today()
    days  = [(today - timedelta(days=i)) for i in range(6, -1, -1)]  # Mon→Sun order

    # ── Try DB records first (real predictions) ──────────────────────────────
    result = {}
    for d in days:
        records = MealRecord.query.filter_by(record_date=d).all()
        if records:
            total = sum(
                r.actual_students if r.actual_students else (r.predicted_students or 0)
                for r in records
            )
            result[d.isoformat()] = total
        else:
            result[d.isoformat()] = None   # no data for this day yet

    # ── Fill missing days from dataset averages per day-of-week ──────────────
    BASE_DIR   = os.path.join(os.path.dirname(__file__), "ml")
    SEED_PATH  = os.path.join(BASE_DIR, "meal_dataset.xlsx")
    LOG_PATH   = os.path.join(BASE_DIR, "actuals_log.xlsx")

    df = None
    try:
        df_seed = pd.read_excel(SEED_PATH, engine="openpyxl")
        if os.path.exists(LOG_PATH):
            df_log = pd.read_excel(LOG_PATH, engine="openpyxl")
            df = pd.concat([df_seed, df_log[["Day_of_Week","Meal_Type","Food_Item",
                                              "Popularity_Index","Expected_Students"]]],
                           ignore_index=True)
        else:
            df = df_seed
    except Exception:
        pass

    day_averages = {}
    if df is not None:
        grp = df.groupby("Day_of_Week")["Expected_Students"].mean()
        day_averages = grp.to_dict()

    # ── Most demanded meal type per day-of-week from dataset ────────────────
    meal_by_day = {}
    if df is not None:
        grp = df.groupby(["Day_of_Week", "Meal_Type"])["Expected_Students"].mean()
        for (day_name, meal), avg in grp.items():
            if day_name not in meal_by_day or avg > meal_by_day[day_name][1]:
                meal_by_day[day_name] = (meal, avg)

    chart_data = []
    for d in days:
        day_name  = d.strftime("%A")
        short_day = d.strftime("%a")
        actual    = result.get(d.isoformat())

        if actual is not None:
            students = actual
            source   = "actual"
        elif day_name in day_averages:
            students = int(day_averages[day_name])
            source   = "average"
        else:
            students = 0
            source   = "none"

        top_meal = meal_by_day.get(day_name, ("—", 0))[0]

        chart_data.append({
            "date":     d.isoformat(),
            "day":      short_day,
            "day_full": day_name,
            "students": students,
            "source":   source,
            "top_meal": top_meal,   # most demanded meal type this day
        })

    # Normalise heights as % of max for the bar chart
    max_val = max((d["students"] for d in chart_data), default=1) or 1
    for d in chart_data:
        d["height_pct"] = round(d["students"] / max_val * 100, 1)

    return jsonify({"chart": chart_data, "max_students": max_val}), 200


@app.route('/api/dashboard/stats', methods=['GET'])
def dashboard_stats():
    """
    Returns real metrics for the Dashboard cards:
    - total_predicted_ever   : sum of all predicted_students in DB
    - avg_expected_students  : average Expected_Students from the dataset
    - total_records          : total number of prediction records
    - most_predicted_food    : food item with highest total predicted students
    """
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    import pandas as pd
    from sqlalchemy import func

    # ── Load full dataset (seed + actuals log) ───────────────────────────────
    BASE_DIR  = os.path.join(os.path.dirname(__file__), "ml")
    SEED_PATH = os.path.join(BASE_DIR, "meal_dataset.xlsx")
    LOG_PATH  = os.path.join(BASE_DIR, "actuals_log.xlsx")

    df = None
    try:
        df = pd.read_excel(SEED_PATH, engine="openpyxl")
        if os.path.exists(LOG_PATH):
            df_log = pd.read_excel(LOG_PATH, engine="openpyxl")
            df = pd.concat([df, df_log[["Day_of_Week","Meal_Type","Food_Item",
                                        "Popularity_Index","Expected_Students"]]],
                           ignore_index=True)
    except Exception:
        df = None

    # ── Total predicted students (dataset + DB predictions) ──────────────────
    # Sum from dataset + any real predictions the user has generated
    dataset_total = int(df["Expected_Students"].sum()) if df is not None else 0
    db_total = int(db.session.query(
        func.sum(MealRecord.predicted_students)
    ).scalar() or 0)
    total_predicted = dataset_total + db_total

    # ── How many times user clicked Generate Prediction ──────────────────────
    total_records = MealRecord.query.count()

    # ── Average expected students (from full dataset) ─────────────────────────
    avg_expected = int(df["Expected_Students"].mean()) if df is not None else 0

    # ── Most predicted food from dataset ─────────────────────────────────────
    most_predicted_food  = "—"
    most_predicted_total = 0
    if df is not None:
        food_totals = df.groupby("Food_Item")["Expected_Students"].sum()
        top_food    = food_totals.idxmax()
        most_predicted_food  = top_food
        most_predicted_total = int(food_totals[top_food])

    # ── Peak day of week (highest avg demand across full dataset) ─────────────
    peak_day = "—"
    if df is not None:
        day_avg  = df.groupby("Day_of_Week")["Expected_Students"].mean()
        peak_day = day_avg.idxmax()

    # ── Most demanded meal type (Breakfast / Lunch / Supper) ──────────────────
    most_demanded_meal = "—"
    if df is not None:
        meal_totals        = df.groupby("Meal_Type")["Expected_Students"].sum()
        most_demanded_meal = meal_totals.idxmax()   # e.g. "Lunch"

    return jsonify({
        "total_predicted_ever":  total_predicted,
        "avg_expected_students": avg_expected,
        "total_records":         total_records,
        "most_predicted_food":   most_predicted_food,
        "most_predicted_total":  most_predicted_total,
        "peak_day":              peak_day,
        "most_demanded_meal":    most_demanded_meal,
    }), 200


@app.route('/api/reports', methods=['GET'])
def get_reports():
    """
    Returns paginated prediction records for the Reports page.
    Query params:
        page     — page number (default 1)
        per_page — rows per page (default 10)
    """
    if 'user_id' not in session:
        return jsonify({"error": "Unauthorized"}), 401

    page     = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 10))

    total   = MealRecord.query.count()
    records = (
        MealRecord.query
        .order_by(MealRecord.id.desc())   # most recent first
        .offset((page - 1) * per_page)
        .limit(per_page)
        .all()
    )

    # Summary stats
    from sqlalchemy import func
    total_predictions = MealRecord.query.count()
    today_str = __import__('datetime').date.today().isoformat()
    completed_today = MealRecord.query.filter(
        MealRecord.record_date == today_str
    ).count()

    return jsonify({
        "records":     [r.to_dict() for r in records],
        "total":       total,
        "page":        page,
        "per_page":    per_page,
        "total_pages": max(1, -(-total // per_page)),   # ceiling division
        "stats": {
            "total_predictions": total_predictions,
            "completed_today":   completed_today,
        }
    }), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)