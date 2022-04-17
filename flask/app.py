from flask import Flask, request, render_template
from model_init import SentimentModel
from model_init import CyberBullyModel

app = Flask(__name__)

saa6124 = SentimentModel('saa6124')
mvg5906= CyberBullyModel('mvg5906')

def get_sentiment(text):
    sentiment_saa6124 = saa6124.get_sentiment(text)

    #TODO: get other 2 sentiment model output & average results
    sentiment = sentiment_saa6124

    return sentiment


def get_cyberbullying(text):
    # TODO
    cyberbullying = mvg5906.get_ctype(text)
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
