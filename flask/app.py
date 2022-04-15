from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# load vectorizers
vectorizer_saa6124 = pickle.load(open('..\\models\\vectorizer_saa6124.pickle', 'rb'))

# load models
#clas=pickle.load(open('clf.plk', 'rb'))
sentiment_model_saa6124 = pickle.load(open('..\\models\\sentiment_saa6124.pickle', 'rb'))

def get_sentiment(text):
    text_vec_saa6124 = vectorizer_saa6124.transform([text])
    prediction_prob_saa6124 = sentiment_model_saa6124.predict_proba(text_vec_saa6124)
    prob_neg_saa6124 = prediction_prob_saa6124[0][0]
    prob_pos_saa6124 = prediction_prob_saa6124[0][1]
    # TODO: get predict_proba for other 2 sentiment models and replace below
    prob_neg_avg = (prob_neg_saa6124 + prob_neg_saa6124 + prob_neg_saa6124) / 3
    prob_pos_avg = (prob_pos_saa6124 + prob_pos_saa6124 + prob_pos_saa6124) / 3
    if (prob_pos_avg > prob_neg_avg):
        sentiment = {
            "value": 1,
            "label": "Positive",
            "probability": round(prob_pos_avg * 100, 2)
        }
    else:
        sentiment = {
            "value": 0,
            "label": "Negative",
            "probability": round(prob_neg_avg * 100, 2)
        }
    return sentiment

def get_cyberbullying(text):
    # TODO
    cyberbullying = {
        "value": 0,
        "label": "Not present",
        "type": "N/A",
        "probability": 100.00
    }
    return cyberbullying

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    text = request.form["tweet_text"]
    sentiment = get_sentiment(text)
    cyberbullying = get_cyberbullying(text)
    return render_template("result.html", text=text, sentiment=sentiment, cyberbullying=cyberbullying)

app.run(debug=True)