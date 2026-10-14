from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model va cac thanh phan
saved = joblib.load("rf_diabetes.pkl")

rf = saved["model"]
scaler = saved["scaler"]
model_columns = saved["model_columns"]
reference_means = saved["reference_means"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():
    # Lay du lieu tu form
    form_data = {
        "gender": request.form["gender"],
        "age": float(request.form["age"]),
        "race": request.form["race"],
        "hypertension": int(request.form["hypertension"]),
        "heart_disease": int(request.form["heart_disease"]),
        "smoking_history": request.form["smoking_history"],
        "bmi": float(request.form["bmi"]),
        "hbA1c_level": float(request.form["hbA1c_level"]),
        "blood_glucose_level": float(
            request.form["blood_glucose_level"]
        )
    }

    # Tao DataFrame dau vao
    X = pd.DataFrame([{
        "gender": form_data["gender"],
        "age": form_data["age"],
        "race:AfricanAmerican": int(
            form_data["race"] == "AfricanAmerican"
        ),
        "race:Asian": int(
            form_data["race"] == "Asian"
        ),
        "race:Caucasian": int(
            form_data["race"] == "Caucasian"
        ),
        "race:Hispanic": int(
            form_data["race"] == "Hispanic"
        ),
        "race:Other": int(
            form_data["race"] == "Other"
        ),
        "hypertension": form_data["hypertension"],
        "heart_disease": form_data["heart_disease"],
        "smoking_history": form_data["smoking_history"],
        "bmi": form_data["bmi"],
        "hbA1c_level": form_data["hbA1c_level"],
        "blood_glucose_level": form_data["blood_glucose_level"]
    }])

    # One-hot encoding
    X = pd.get_dummies(
        X,
        columns=["gender", "smoking_history"],
        dtype=int
    )

    # Dong bo cot voi du lieu huan luyen
    X = X.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Chuan hoa
    X_scaled = scaler.transform(X)

    # Du doan
    prediction = int(rf.predict(X_scaled)[0])

    # Xac suat mo hinh du doan thuoc lop 1
    probabilities = rf.predict_proba(X_scaled)[0]
    class_index = list(rf.classes_).index(prediction)
    confidence = float(probabilities[class_index]) * 100
    confidence = round(confidence, 2)

    # Mau sac theo ket qua
    confidence_color = (
        "#dc2626" if prediction == 1 else "#16a34a"
    )

    # So sanh voi gia tri tham chieu
    advice = []

    for feature in [
        "bmi",
        "hbA1c_level",
        "blood_glucose_level",
        "age"
    ]:
        value = form_data[feature]
        reference = float(reference_means[feature])

        if reference == 0:
            continue

        difference = (value - reference) / reference

        if abs(difference) >= 0.1:
            status = "high" if difference > 0 else "low"

            advice.append({
                "feature": feature,
                "status": status,
                "value": round(value, 2),
                "reference": round(reference, 2),
                "message": (
                    "Giá trị cao hơn mức tham chiếu."
                    if status == "high"
                    else "Giá trị thấp hơn mức tham chiếu."
                )
            })

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=confidence,
        confidence_color=confidence_color,
        form_data=form_data,
        advice=advice
    )


if __name__ == "__main__":
    app.run(debug=True)