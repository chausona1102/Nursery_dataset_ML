from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key'

model = joblib.load("decision_tree_best.pkl")
encoders = joblib.load("label_encoders_best.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        fields = ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social', 'health']
        input_data = []
        values = {}
        try:
            for field in fields:
                raw = request.form.get(field)
                values[field] = raw
                encoded = encoders[field].transform([raw])[0]
                input_data.append(encoded)
                
            input_data = np.array([input_data])
            pred_encoded = model.predict(input_data)[0]
            pred_lable = encoders["class"].inverse_transform([pred_encoded])[0]
            
            session["values"] = values
            session["prediction"] = pred_lable
            return redirect(url_for("index"))
            
        except Exception as e:
            return render_template("index.html", error = str(e), prediction=None, values=request.form)
            
    prediction = session.pop("prediction", None)
    values = session.pop("values", {})
    return render_template("index.html", prediction=prediction, error=None, values=values)

if __name__ == "__main__":
    app.run(debug=True)