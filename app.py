from flask import Flask, request, render_template
from model_init import ModelWithVectorizer, SequentialModel, CyberBullyModel

app = Flask(__name__)

saa6124 = ModelWithVectorizer('saa6124')
cpb5703 = ModelWithVectorizer('cpb5703')
aav5195 = SequentialModel('aav5195')
mvg5906 = CyberBullyModel('mvg5906')
mvg5906_sgd = CyberBullyModel('sgdmvg5906')

def get_sentiment(text):
    sentiment_saa6124 = saa6124.get_sentiment(text)
    sentiment_cpb5703 = cpb5703.get_sentiment(text)
    sentiment_aav5195 = aav5195.get_sequential_sentiment(text)

    #TODO: get other 2 sentiment model output & average results
    #sentiment = sentiment_saa6124
    sentiment = sentiment_cpb5703
    sentimentSequential = sentiment_aav5195
    return sentimentSequential
    #return sentiment


def get_cyberbullying(text):
    # TODO
    #cyberbullying1 = mvg5906.get_ctype(text)
    #yberbullying2 = mvg5906_sgd.get_ctype(text)
    #ensemble_cyber = (cyberbullying1 + cyberbullying2)/2
    cyberbullying = mvg5906_sgd.get_ctype(text)
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
