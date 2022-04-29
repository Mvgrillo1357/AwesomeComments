from flask import Flask, request, render_template
from models.model_init import Model, sentiment_labels, cyberbullying_labels
from models.ensemble import get_ensemble_prediction

# Initialize flask application
app = Flask(__name__)

# Create sentiment model ensemble
sentiment_models = [
    Model('sentiment_saa6124.pickle',
          'vectorizer_saa6124.pickle'),
    Model('sentiment_cpb5703.pickle',
          'vectorizer_cpb5703.pickle'),
    Model('sentiment_aav5195.pickle',
          'vectorizer_aav5195.pickle')
]

# Create cyberbullying model ensemble 
cyberbullying_models = [
    Model('cyberbullysentiment_sgdmvg5906.pickle',
          'cyberbullyvectorizer_sgdmvg5906.pickle'),
    Model('cyberbullysentiment_dtreemvg5906.pickle',
          'cyberbullyvectorizer_dtreemvg5906.pickle'),
    Model('cyberbullysentiment_mlpclassmvg5906.pickle',
          'cyberbullyvectorizer_mlpclassmvg5906.pickle')
]


def get_sentiment_prediction(text):
    # Get emsemble result for sentiment model collection
    prediction = get_ensemble_prediction(sentiment_models, text)
    # For display purposes, multiply and round the prediction probability
    prediction["probability"] = round(prediction["probability"] * 100, 2)
    # Add the prediction class display label to the result
    prediction["label"] = sentiment_labels[prediction["class"]]
    return prediction


def get_cyberbullying_prediction(text):
    # Get emsemble result for cyberbullying model collection
    prediction = get_ensemble_prediction(cyberbullying_models, text)
    # For display purposes, multiply and round the prediction probability
    prediction["probability"] = round(prediction["probability"] * 100, 2)
    # Add the prediction class display label to the result
    prediction["label"] = cyberbullying_labels[prediction["class"]]
    return prediction


# Handler for root requests, returns layout template with empty form
@ app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# Handler for form submission, returns layout with result container
@ app.route("/result", methods=["POST"])
def result():
    # Get the input text from the request form
    text = request.form["tweet_text"]
    # Get the sentiment prediction
    sentiment = get_sentiment_prediction(text)
    # Get the cyberbullying prediction
    cyberbullying = get_cyberbullying_prediction(text)
    # Return the view template with the prediction results
    return render_template("result.html", text=text, sentiment=sentiment, cyberbullying=cyberbullying)


# Run flask application
app.run()
