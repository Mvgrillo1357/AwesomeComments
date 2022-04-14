from flask import Flask, request, url_for, render_template
import pickle

#load clf model
clas=pickle.load(open('clf.plk', 'rb'))
clf1= pickle.load(open('sentiment_mnb_vec.pickle', 'rb'))



app = Flask(__name__)

# When a GET request is sent to the context "/"
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():
    text= request.form["tweet_text"]
    cyberbullying= request.form["cyberbullying_type"]
    sentiment = request.form["cyberbullying_type"]
    X_test_pkl=request.form["sentiment"]
    cyberbullying=clas.predict(X_test_pkl)
    sentiment= clf1.predict_proba(X_test_pkl)
    return render_template("result.html", text= text, cyberbullying=cyberbullying, sentiment= sentiment)

app.run(debug=True)