from flask import Flask, request, url_for, render_template
import pickle

#load vectorizers
vectorizer_saa6124 = pickle.load(open('..\\models\\vectorizer_saa6124.pickle', 'rb'))

#load models
#clas=pickle.load(open('clf.plk', 'rb'))
sentiment_model_saa6124 = pickle.load(open('..\\models\\sentiment_saa6124.pickle', 'rb'))

app = Flask(__name__)

# When a GET request is sent to the context "/"
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["tweet_text"]
    text_vec = vectorizer_saa6124.transform([text])
    sentiment_saa6124 = sentiment_model_saa6124.predict(text_vec)
    # cyberbullying= request.form["cyberbullying_type"]
    # sentiment = request.form["cyberbullying_type"]
    # X_test_pkl=request.form["sentiment"]
    #cyberbullying=clas.predict(X_test_pkl)
    # sentiment= clf1.predict_proba(X_test_pkl)
    #return render_template("result.html", text= text, cyberbullying=cyberbullying, sentiment = sentiment)
    return render_template("result.html", text = text, sentiment = sentiment_saa6124)

app.run(debug=True)