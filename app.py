from flask import Flask, request, render_template
from model_init import SentimentModel, SequentialModel, CyberBullyModel, sentiment_labels, cyberbully_labels

app = Flask(__name__)

saa6124 = SentimentModel('saa6124')
cpb5703 = SentimentModel('cpb5703')
#aav5195 = SequentialModel('aav5195')
mvg5906 = CyberBullyModel('mvg5906')
mvg5906_sgd = CyberBullyModel('sgdmvg5906')

def get_ensemble_prediction(probabilities, labels):
    probabilities_ensemble = sum(probabilities)
    probabilities_ensemble = probabilities_ensemble / probabilities_ensemble.size
    probability = probabilities_ensemble.max()
    prediction = probabilities_ensemble.tolist().index(probability)
    label = labels[prediction]
    return {
        "value": prediction,
        "label": label,
        "probability": round(probability * 100, 2)
    }


def get_sentiment(text):
    probabilities_saa6124 = saa6124.get_probabilities(text)
    probabilities_cpb5703 = cpb5703.get_probabilities(text)
    #sentiment_aav5195 = aav5195.get_sequential_sentiment(text)
    return get_ensemble_prediction(
        [probabilities_saa6124, probabilities_cpb5703], 
        sentiment_labels)


def get_ensemble_cyberbullying(probabilities, labels):
    probabilities_ensemble = sum(probabilities)
    probabilities_ensemble = probabilities_ensemble / probabilities_ensemble.size
    probability = probabilities_ensemble.max()
    prediction = probabilities_ensemble.tolist().index(probability)
    label = labels[prediction]
    return {
        "value": prediction,
        "label": label,
        "probability": round(probabilities.max() *100, 2)
    }
def get_sentiment_cyber(text):
    probabilities_mvg5906 = mvg5906.get_probabilities(text)
    probabilities_mvg5906_sgd = mvg5906_sgd.get_probabilities(text)
    return get_ensemble_prediction(
        [probabilities_mvg5906, probabilities_mvg5906_sgd], 
        cyberbully_labels)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():
    text = request.form["tweet_text"]
    sentiment = get_sentiment(text)
    cyberbullying = get_sentiment_cyber(text)
    return render_template("result.html", text=text, sentiment=sentiment, cyberbullying=cyberbullying)


app.run(debug=True)
