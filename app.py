from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# =========================
# Load model, scaler, reference values
# =========================

saved = joblib.load("rf_diabetes.pkl")

model = saved["model"]
scaler = saved["scaler"]
FEATURES = saved["feature_names"]
reference_means = saved["reference_means"]

# =========================
# Generate advice
# =========================

def gen_advice(values, reference_means):

    advice = []

    advice_text = {

        "age": {
            "high": "Bạn nên duy trì lối sống lành mạnh và kiểm tra sức khỏe định kỳ.",
            "low": "Hãy tiếp tục duy trì chế độ ăn uống và vận động hợp lý."
        },

        "bmi": {
            "high": (
                "Chỉ số BMI của bạn cao hơn mức tham chiếu của nhóm không mắc "
                "tiểu đường. Bạn nên chú ý duy trì chế độ ăn uống cân bằng và "
                "hoạt động thể chất phù hợp."
            ),
            "low": (
                "Chỉ số BMI của bạn thấp hơn mức tham chiếu của nhóm không mắc "
                "tiểu đường. Bạn nên đảm bảo chế độ ăn uống đầy đủ và cân bằng."
            )
        },

        "HbA1c_level": {
            "high": (
                "Chỉ số HbA1c của bạn cao hơn mức tham chiếu trong tập dữ liệu. "
                "Bạn nên chú ý đến chế độ ăn uống, vận động và theo dõi sức khỏe định kỳ."
            ),
            "low": (
                "Chỉ số HbA1c của bạn thấp hơn mức tham chiếu trong tập dữ liệu. "
                "Hãy tiếp tục duy trì chế độ sinh hoạt và ăn uống cân bằng."
            )
        },

        "blood_glucose_level": {
            "high": (
                "Nồng độ glucose trong máu của bạn cao hơn mức tham chiếu trong "
                "tập dữ liệu. Bạn nên chú ý đến chế độ ăn uống và theo dõi sức khỏe định kỳ."
            ),
            "low": (
                "Nồng độ glucose trong máu của bạn thấp hơn mức tham chiếu trong "
                "tập dữ liệu. Nên duy trì chế độ ăn uống đều đặn và cân bằng."
            )
        }
    }


    # Duyệt trực tiếp dictionary
    for feature, value in values.items():

        # Feature không có trong reference
        if feature not in reference_means:
            continue

        # Feature không có lời khuyên
        if feature not in advice_text:
            continue


        mean_value = reference_means[feature]


        # Cao hơn 10%
        if value > mean_value * 1.10:

            advice.append({
                "feature": feature,
                "value": round(value, 2),
                "reference": round(mean_value, 2),
                "status": "high",
                "message": advice_text[feature]["high"]
            })


        # Thấp hơn 10%
        elif value < mean_value * 0.90:

            advice.append({
                "feature": feature,
                "value": round(value, 2),
                "reference": round(mean_value, 2),
                "status": "low",
                "message": advice_text[feature]["low"]
            })


    return advice

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return redirect(url_for("home"))


    # =========================
    # Get input from form
    # =========================

    features = {
        "gender": request.form["gender"],

        "age": float(request.form["age"]),

        "hypertension": int(request.form["hypertension"]),

        "heart_disease": int(request.form["heart_disease"]),

        "smoking_history": request.form["smoking_history"],

        "bmi": float(request.form["bmi"]),

        "HbA1c_level": float(request.form["HbA1c_level"]),

        "blood_glucose_level": float(
            request.form["blood_glucose_level"]
        )
    }


    # =========================
    # Convert to DataFrame
    # =========================

    input_df = pd.DataFrame([features])


    # =========================
    # One-Hot Encoding
    # =========================

    input_encoded = pd.get_dummies(input_df)


    # =========================
    # Match features used in training
    # =========================

    input_encoded = input_encoded.reindex(
        columns=FEATURES,
        fill_value=0
    )


    # =========================
    # Scale
    # =========================

    x_scaled = scaler.transform(input_encoded)


    # =========================
    # Predict
    # =========================

    prediction = model.predict(x_scaled)[0]

    probabilities = model.predict_proba(x_scaled)[0]

    confidence = probabilities[int(prediction)] * 100


    # =========================
    # Confidence color
    # =========================

    if confidence >= 80:
        confidence_color = "green"

    elif confidence >= 60:
        confidence_color = "orange"

    else:
        confidence_color = "red"


    # =========================
    # Generate advice
    # =========================

    advice = gen_advice(
        features,
        reference_means
    )


    # =========================
    # Return result
    # =========================

    return render_template(
        "result.html",

        form_data=features,

        prediction=prediction,

        confidence=round(confidence, 2),

        confidence_color=confidence_color,

        advice=advice
    )


# =========================
# Run app
# =========================

if __name__ == "__main__":
    app.run(debug=True)