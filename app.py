from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# nhập model vs scaler
model = joblib.load("knn_diabetes.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return redirect(url_for("home"))

    # đặc trưng từ form
    features = [
        int(request.form["Pregnancies"]),
        int(request.form["Glucose"]),
        int(request.form["BloodPressure"]),
        int(request.form["SkinThickness"]),
        int(request.form["Insulin"]),
        float(request.form["BMI"]),
        float(request.form["DiabetesPedigreeFunction"]),
        int(request.form["Age"])
    ]

    #shape về arr 1x8 để so mẫu trong model đã train
    x = np.array(features).reshape(1, -1)
    x_scaled = scaler.transform(x)

    prediction = model.predict(x_scaled)[0]
    probabilities = model.predict_proba(x_scaled)[0]
    confidence = probabilities[prediction] * 100

    confidence_color = "red"
    if confidence >= 80:
        confidence_color = "green"
    elif confidence >= 60:
        confidence_color = "orange"

    return render_template(
        "result.html",
        form_data=request.form,
        prediction=prediction,
        confidence=confidence,
        confidence_color=confidence_color
    )


if __name__ == "__main__":
    app.run(debug=True)
    