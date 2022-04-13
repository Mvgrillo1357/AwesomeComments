from flask import Flask, request, url_for, render_template
import pickle

#load clf model
clas=pickle.load(open('clf.plk', 'rb'))

#Use a Preprocessor to input lat, lon




app = Flask(__name__)

# When a GET request is sent to the context "/"
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():
    text= request.form["tweet_text"]
    latitude= request.form["cyberbullying_type"]
    X_test_pkl=[[tweet_text, cyberbullying_type]]
    severity=clas.predict(X_test_pkl)
    return render_template("result.html", lng= longitude, lat=latitude, sev= severity)

app.run(debug=True)