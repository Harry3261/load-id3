from flask import Flask, jsonify, render_template, request

import id3

app = Flask(__name__)

TREE = id3.train()
OPTIONS = {
    "Income": ["Low", "Medium", "High"],
    "Employment": ["Employed", "Self-Employed", "Unemployed"],
    "CreditHistory": ["Good", "Average", "Poor"],
}


@app.route("/")
def index():
    return render_template("index.html", options=OPTIONS)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    sample = {f: data.get(f) for f in id3.FEATURES}
    for f, value in sample.items():
        if value not in OPTIONS[f]:
            return jsonify(error=f"Invalid or missing value for {f}"), 400
    label, path = id3.predict(TREE, sample)
    return jsonify(
        eligible=(label == "Yes"),
        result="Eligible" if label == "Yes" else "Not Eligible",
        path=path,
    )


@app.route("/tree")
def tree():
    gains = {
        f: round(id3.info_gain(id3.rows_as_dicts(), f), 4) for f in id3.FEATURES
    }
    return jsonify(tree=TREE, gains=gains, training_rows=len(id3.DATA))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
