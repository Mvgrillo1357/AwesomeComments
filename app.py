from flask import Flask, request, render_template
from models.model_init import Model, sentiment_labels, cyberbullying_labels
from models.ensemble import get_ensemble_prediction
from services.twitter import get_tweet_conversation

app = Flask(__name__)

saa6124 = Model('sentiment_saa6124.pickle',
                'vectorizer_saa6124.pickle')
cpb5703 = Model('sentiment_cpb5703.pickle',
                'vectorizer_cpb5703.pickle')
mvg5906 = Model('cyberbullysentiment_mvg5906.pickle',
                'cyberbullyvectorizer_mvg5906.pickle')
mvg5906_sgd = Model('cyberbullysentiment_sgdmvg5906.pickle',
                    'cyberbullyvectorizer_sgdmvg5906.pickle')
#aav5195 = SequentialModel('aav5195')


def get_sentiment_prediction(text):
    prediction = get_ensemble_prediction([saa6124, cpb5703], text)
    prediction["label"] = sentiment_labels[prediction["class"]]
    return prediction


def get_cyberbullying_prediction(text):
    prediction = get_ensemble_prediction([mvg5906, mvg5906_sgd], text)
    prediction["label"] = cyberbullying_labels[prediction["class"]]
    return prediction


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():
    tweet_url = request.form["tweet_url"]
    tweet_conversation = get_tweet_conversation(tweet_url)
    for tweet in tweet_conversation:
        tweet["sentiment"] = get_sentiment_prediction(tweet["text"])
        tweet["cyberbullying"] = get_cyberbullying_prediction(tweet["text"])
    return render_template("result.html", tweet_url=tweet_url, tweets=tweet_conversation)


app.run(debug=True)
