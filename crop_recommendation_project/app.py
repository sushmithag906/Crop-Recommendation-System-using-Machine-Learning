from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("crop_recommendation_model.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    values = {}

    if request.method == "POST":
        values = {
            "N": request.form["N"],
            "P": request.form["P"],
            "K": request.form["K"],
            "temperature": request.form["temperature"],
            "humidity": request.form["humidity"],
            "ph": request.form["ph"],
            "rainfall": request.form["rainfall"]
        }

        input_data = np.array([[
            float(values["N"]),
            float(values["P"]),
            float(values["K"]),
            float(values["temperature"]),
            float(values["humidity"]),
            float(values["ph"]),
            float(values["rainfall"])
        ]])

        prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction=prediction,
        values=values
    )

if __name__ == "__main__":
    app.run(debug=True)
