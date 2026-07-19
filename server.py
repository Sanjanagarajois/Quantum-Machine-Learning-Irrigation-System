print("[STARTUP] Importing libraries...")

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pennylane as qml
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import pandas as pd

print("[OK] Libraries imported successfully")

EXCEL_FILE = "training_data.xlsx"
FEATURE_COLS = [
    "soil_moisture", "temperature", "humidity", "light",
    "nitrogen", "phosphorus", "potassium"
]
LABEL_COL = "label"
N_FEATURES = len(FEATURE_COLS)
DEV = qml.device("default.qubit", wires=N_FEATURES)
print(f"[SETUP] Quantum simulator ready with {N_FEATURES} qubits")

RAW_RANGES = {
    "soil_moisture": (0, 100),      
    "temperature": (20, 45),        
    "humidity": (0, 100),           
    "light": (0, 100000),          
    "nitrogen": (0, 200),       
    "phosphorus": (0, 200),         
    "potassium": (0, 200),          
}
GROUNDNUT_IDEAL = {
    "nitrogen":   (0.40, 0.60),
    "phosphorus": (0.30, 0.50),
    "potassium":  (0.40, 0.60)
}

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def normalize_raw(value, feature_name):
    lo, hi = RAW_RANGES[feature_name]
    if hi == lo:
        return 0.0
    return clamp01((float(value) - lo) / (hi - lo))

def auto_read_value(d, key):
    if key in d:
        norm = clamp01(d[key])
        lo, hi = RAW_RANGES[key]
        raw_est = lo + norm * (hi - lo)
        return raw_est, norm
    raw_key = key + "_raw"
    if raw_key in d:
        raw_val = float(d[raw_key])
        norm = normalize_raw(raw_val, key)
        return raw_val, norm
    return None, None

@qml.qnode(DEV)
def quantum_feature_map(x):
    for i in range(N_FEATURES):
        qml.RY(x[i], wires=i)  
    for i in range(N_FEATURES - 1):
        qml.CNOT(wires=[i, i + 1]) 
    return [qml.expval(qml.PauliZ(i)) for i in range(N_FEATURES)] 

def quantum_transform(X):
    return np.array([quantum_feature_map(row) for row in X]) 

print("Quantum feature map created")

def load_dataset():
    print(f"[DATA] Loading training dataset from: {EXCEL_FILE}")
    if not os.path.exists(EXCEL_FILE):
        raise FileNotFoundError(
            f"Missing {EXCEL_FILE} inside backend folder.\n"
            f"Create it with columns: {FEATURE_COLS + [LABEL_COL]}"
        )
    df = pd.read_excel(EXCEL_FILE)
    for c in FEATURE_COLS + [LABEL_COL]:
        if c not in df.columns:
            raise ValueError(f"Excel missing column: {c}")
    df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df[LABEL_COL].astype(str).to_numpy()
    if len(X) < 4:
        raise ValueError("Dataset must contain at least 4 valid rows.")
    print(f"[OK] Loaded {len(X)} rows from Excel")
    return X, y

X_train, y_train = load_dataset()
print("[TRAINING] Quantum transform training set...")
Xq_train = quantum_transform(X_train)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y_train)

print(f"[INFO] Label mapping: { {label: idx for idx, label in enumerate(le.classes_)} }")
Xq_tr, Xq_val, y_tr, y_val = train_test_split(
    Xq_train, y_encoded, test_size=0.2, random_state=42,
    stratify=y_encoded if len(set(y_encoded)) > 1 else None
)

print("[TRAINING] Training neural network...")
model = Pipeline([
    ("scaler", StandardScaler()),
    ("nn", MLPClassifier(
        hidden_layer_sizes=(32, 24, 16, 8),
        activation="relu",
        solver="adam",
        alpha=0.001,
        max_iter=1000,
        random_state=42,
        early_stopping=True,        
        validation_fraction=0.1,
        verbose=False
    ))
])
model.fit(Xq_tr, y_tr)
val_acc = model.score(Xq_val, y_val)
print(f"[OK] Neural network trained. Validation accuracy: {val_acc*100:.1f}%")
print("[OK] Architecture: 7 → 32 → 24 → 16 → 8 → output classes")

def nutrient_status(value, low, high):
    if value < low:
        return "LOW"
    if value > high:
        return "HIGH"
    return "OPTIMAL"

def generate_nutrient_report(n, p, k):
    n_low, n_high = GROUNDNUT_IDEAL["nitrogen"]
    p_low, p_high = GROUNDNUT_IDEAL["phosphorus"]
    k_low, k_high = GROUNDNUT_IDEAL["potassium"]
    report = {
        "nitrogen": {
            "value_normalized": round(n, 3),
            "ideal_range_normalized": [n_low, n_high],
            "status": nutrient_status(n, n_low, n_high)
        },
        "phosphorus": {
            "value_normalized": round(p, 3),
            "ideal_range_normalized": [p_low, p_high],
            "status": nutrient_status(p, p_low, p_high)
        },
        "potassium": {
            "value_normalized": round(k, 3),
            "ideal_range_normalized": [k_low, k_high],
            "status": nutrient_status(k, k_low, k_high)
        }
    }
    return report

def fertilizer_advice(n_report):
    adv = []
    if n_report["nitrogen"]["status"] == "LOW":
        adv.append("Nitrogen LOW → apply Nitrogen fertilizer (Urea/organic manure).")
    elif n_report["nitrogen"]["status"] == "HIGH":
        adv.append("Nitrogen HIGH → avoid extra nitrogen fertilizer.")
    if n_report["phosphorus"]["status"] == "LOW":
        adv.append("Phosphorus LOW → apply phosphate fertilizer (DAP/SSP).")
    elif n_report["phosphorus"]["status"] == "HIGH":
        adv.append("Phosphorus HIGH → avoid extra phosphate.")
    if n_report["potassium"]["status"] == "LOW":
        adv.append("Potassium LOW → apply potash fertilizer (MOP).")
    elif n_report["potassium"]["status"] == "HIGH":
        adv.append("Potassium HIGH → avoid extra potash.")
    if not adv:
        adv.append("NPK OPTIMAL → no fertilizer needed now.")
    return adv

def decide_pump(status, soil_norm):
    if status == "NEEDS_WATER":
        return "ON"                       
    if status == "HEAT_STRESS" and soil_norm < 0.65:
        return "ON"                         
    if status == "FROST_STRESS":
        return "ON"                        
    if status == "WATERLOGGED":
        return "OFF"                        
    if status == "PEST_DAMAGE" and soil_norm < 0.40:
        return "ON"                         
    return "OFF"

def irrigation_advice(status, soil_norm, temp_norm):
    adv = []
    if soil_norm < 0.30:
        adv.append("Soil moisture very low → irrigate immediately.")
    elif soil_norm < 0.45:
        adv.append("Soil moisture low → irrigate soon.")
    if temp_norm > 0.85:
        adv.append("Temperature very high → irrigate early morning or evening.")
    if status == "HEALTHY" and soil_norm >= 0.45:
        adv.append("Soil moisture sufficient → monitor only.")
    elif status == "WATERLOGGED":
        adv.append("Soil is waterlogged → stop all irrigation immediately.")
        adv.append("Improve drainage — open furrows or raise beds if possible.")
        adv.append("Do NOT irrigate until soil moisture drops below 0.60.")
    elif status == "FROST_STRESS":
        adv.append("Frost detected → irrigate to keep soil warm (wet soil retains heat).")
        adv.append("Water early evening before temperature drops further.")
    elif status == "PEST_DAMAGE":
        adv.append("Pest-damaged plants lose water faster → maintain moisture above 0.45.")
        adv.append("Avoid over-watering — wet leaves worsen fungal secondary infections.")
    elif status == "NUTRIENT_DEFICIENCY":
        adv.append("Consistent moisture helps nutrient absorption — keep soil at 0.45–0.65.")
    return adv if adv else ["No specific irrigation advice."]

app = Flask(__name__)
CORS(app)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "OK", "message": "Quantum Groundnut Backend running"})

@app.route("/monitor", methods=["POST"])
def monitor():
    d = request.json or {}
    mode = str(d.get("mode", "AUTO")).upper()
    raw_vals = {}
    norm_vals = {}
    for key in FEATURE_COLS:
        raw, norm = auto_read_value(d, key)
        if mode == "RAW":
            if raw is None:
                return jsonify({"error": f"Missing raw input: {key}_raw"}), 400
            norm = normalize_raw(raw, key)
        elif mode == "NORM":
            if norm is None:
                return jsonify({"error": f"Missing normalized input: {key}"}), 400
            lo, hi = RAW_RANGES[key]
            raw = lo + norm * (hi - lo)
        else:
            if raw is None and norm is None:
                return jsonify({"error": f"Missing input: provide {key} (normalized) or {key}_raw (raw units)"}), 400
            if norm is None:
                norm = normalize_raw(raw, key)
            if raw is None:
                lo, hi = RAW_RANGES[key]
                raw = lo + norm * (hi - lo)
        raw_vals[key] = round(float(raw), 3)
        norm_vals[key] = round(float(norm), 3)
        
    X_input = np.array([[norm_vals[k] for k in FEATURE_COLS]])
    Xq_input = quantum_transform(X_input)
    predicted_number = model.predict(Xq_input)[0]
    status = le.inverse_transform([predicted_number])[0]  
    
    pump = decide_pump(status, norm_vals["soil_moisture"])
    n_report = generate_nutrient_report(norm_vals["nitrogen"], norm_vals["phosphorus"], norm_vals["potassium"])
    
    response = {
        "plant": "GROUNDNUT",
        "climate": "HOT_AND_DRY",
        "mode_used": mode,
        "inputs_raw_units": {
            "soil_moisture_percent": raw_vals["soil_moisture"],
            "temperature_celsius": raw_vals["temperature"],
            "humidity_percentRH": raw_vals["humidity"],
            "light_lux": raw_vals["light"],
            "nitrogen_mgkg": raw_vals["nitrogen"],
            "phosphorus_mgkg": raw_vals["phosphorus"],
            "potassium_mgkg": raw_vals["potassium"],
        },
        "inputs_normalized_0_to_1": norm_vals,
        "plant_status": status,
        "water_pump": pump,
        "nutrient_report": n_report,
        "irrigation_advice": irrigation_advice(status, norm_vals["soil_moisture"], norm_vals["temperature"]),
        "fertilizer_advice": fertilizer_advice(n_report),
        "summary": (
            f"Neural network classified plant as {status}. "
            f"Pump action: {pump}. "
            f"N: {n_report['nitrogen']['status']}, "
            f"P: {n_report['phosphorus']['status']}, "
            f"K: {n_report['potassium']['status']}."
        )
    }
    return jsonify(response)

if __name__ == "__main__":
    print("[SERVER] Backend running at http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)