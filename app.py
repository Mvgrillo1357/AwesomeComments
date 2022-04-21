from flask import Flask, request, render_template
from models.model_init import Model, sentiment_labels, cyberbullying_labels
from models.ensemble import get_ensemble_prediction

app = Flask(__name__)

saa6124 = Model('sentiment_saa6124.pickle',
                'vectorizer_saa6124.pickle')
cpb5703 = Model('sentiment_cpb5703.pickle',
                'vectorizer_cpb5703.pickle')

aav5195 = Model('sentiment_aav5195.pickle',
                'vectorizer_aav5195.pickle')

#mvg5906 = Model('cyberbullysentiment_mvg5906.pickle','cyberbullyvectorizer_mvg5906.pickle')

mvg5906_sgd = Model('cyberbullysentiment_sgdmvg5906.pickle',
                    'cyberbullyvectorizer_sgdmvg5906.pickle')
mvg5906_dtree = Model('cyberbullysentiment_dtreemvg5906.pickle','cyberbullyvectorizer_dtreemvg5906.pickle')

mvg5906_mlpclass = Model('cyberbullysentiment_mlpclassmvg5906.pickle', 'cyberbullyvectorizer_mlpclassmvg5906.pickle')

#aav5195 = SequentialModel('aav5195')


def get_sentiment_prediction(text):
    prediction = get_ensemble_prediction([saa6124, cpb5703, aav5195], text)
    prediction["label"] = sentiment_labels[prediction["class"]]
    return prediction


def get_cyberbullying_prediction(text):
    prediction = get_ensemble_prediction([mvg5906_sgd, mvg5906_dtree, mvg5906_mlpclass], text)
    prediction["label"] = cyberbullying_labels[prediction["class"]]
    return prediction
    

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/result", methods=["POST"])
def result():
    text = request.form["tweet_text"]
    sentiment = get_sentiment_prediction(text)
    cyberbullying = get_cyberbullying_prediction(text)
    return render_template("result.html", text=text, sentiment=sentiment, cyberbullying=cyberbullying)


app.run(debug=True)
