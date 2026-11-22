
# import the necessary libraries 
import pickle
from flask import Flask, render_template, request

# load the files
with open("stress_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)
# create the app
app = Flask(__name__)

# get the app route
@app.get("/")
def home():
    return render_template("index.html")

# create the prediction route
@app.post("/predict")
def predict():
    text = request.form.get("text", "").strip()
    if not text:
        return render_template("index.html", error="Please enter some text to analyze."), 400

    prediction = str(model.predict([text])[0])
    return render_template("index.html", prediction=prediction, text=text)

# run thew app
if __name__ == "__main__":
    app.run(debug=True)
