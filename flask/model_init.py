import pickle
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences

sentiment_labels = {
    0: "Negative",
    1: "Positive",
    4: "Positive"
}

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

cyberbully_labels = {
    "age" : "Ageism",
    "ethnicity" : "Racism",
    "gender" : "Ageism",
    "not_cyberbullying" : "Not Cyberbullying",
    "other_cyberbullying" : "Another Type of Cyberbullying",
    "religion" : "Anti Religious"
}

class SequentialModel:
    def __init__(self, model_name):
        self.model = keras.models.load_model('..\\sentiment_aav5195')

    def get_sequential_sentiment(self, text):
        tokenizer = Tokenizer()
        # Tokenize text
        x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=300)
        prediction = self.model.predict([x_test])[0]

        # for testing only
        return {
            "value": prediction,
            "label": sentiment_labels[0],
            "probability": round(prediction.max() * 100, 2)
        }
        # return prediction

class SentimentModel:
    def __init__(self, model_name):
        self.vectorizer = pickle.load(open(f'..\\models\\vectorizer_{model_name}.pickle', 'rb'))
        self.model = pickle.load(open(f'..\\models\\sentiment_{model_name}.pickle', 'rb'))

    def get_sentiment(self, text):
        text_vectorized = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vectorized)[0]
        probabilities = self.model.predict_proba(text_vectorized)[0]
        return {
            "value": prediction,
            "label": sentiment_labels[prediction],
            "probability": round(probabilities.max() * 100, 2)
        }



class CyberBullyModel:
    def __init__(self, model_name):
        self.vectorizer = pickle.load(open(f'..\\models\\cyberbullyvectorizer_{model_name}.pickle', 'rb'))
        self.model = pickle.load(open(f'..\\models\\cyberbullysentiment_{model_name}.pickle', 'rb'))

    def get_ctype(self, text):
        text_vectorized = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vectorized)[0]
        probabilities = self.model.predict_proba(text_vectorized)[0]
        return {
            "value": prediction,
            "label": cyberbully_labels[prediction],
            "probability": round(probabilities.max() *100, 2)
        }