from flask import Flask, render_template, request, jsonify
from model import predict

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.json

    input_data = [
        int(data["age"]),
        int(data["cigarettes"]),
        int(data["years"]),
        int(data["income"]),
        int(data["disease"])
    ]

    result = predict(input_data)

    return jsonify({
        "prediction": "High Mortality Risk" if result == 1 else "Low Mortality Risk"
    })

if __name__ == "__main__":
    app.run(debug=True)
