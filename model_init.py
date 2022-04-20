import os
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
    "gender" : "Sexism",
    "not_cyberbullying" : "Not Cyberbullying",
    "other_cyberbullying" : "Another Type of Cyberbullying",
    "religion" : "Anti Religious"
}

class SequentialModel:
    def __init__(self, model_name):
        self.model = keras.models.load_model(os.path.join(os.path.dirname(__file__), 'models', f'sentiment_{model_name}.h5'))

    def get_sequential_sentiment(self, text):
        tokenizer = Tokenizer()
        # Tokenize text
        x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=300)

        prediction = self.model.predict([x_test])[0]
        return {
            "value": prediction,
            "label": sentiment_labels[0],
            "probability": round(prediction.max() * 100, 2)
        }


class SentimentModel:
    def __init__(self, model_name):
        vec_file = os.path.join(os.path.dirname(__file__), 'models', f'vectorizer_{model_name}.pickle')
        self.vectorizer = pickle.load(open(vec_file, 'rb'))
        model_file = os.path.join(os.path.dirname(__file__), 'models', f'sentiment_{model_name}.pickle')
        self.model = pickle.load(open(model_file, 'rb'))

    def get_probabilities(self, text):
        text_vectorized = self.vectorizer.transform([text])
        probabilities = self.model.predict_proba(text_vectorized)[0]
        return probabilities


class CyberBullyModel:
    def __init__(self, model_name1, model_name2):
        vec_file1 = os.path.join(os.path.dirname(__file__), 'models', f'cyberbullyvectorizer_{model_name1}.pickle')
        self.vectorizer = pickle.load(open(vec_file, 'rb'))
        model_file1 = os.path.join(os.path.dirname(__file__), 'models', f'cyberbullysentiment_{model_name1}.pickle')
        self.model = pickle.load(open(model_file, 'rb'))
        vec_file2 = os.path.join(os.path.dirname(__file__), 'models', f'cyberbullyvectorizer_{model_name2}.pickle')
        self.vectorizer = pickle.load(open(vec_file, 'rb'))
        model_file2 = os.path.join(os.path.dirname(__file__), 'models', f'cyberbullysentiment_{model_name2}.pickle')
        self.model = pickle.load(open(model_file, 'rb'))
    def get_ctype(self, text):
        text_vectorized = self.vectorizer.transform([text])
        prediction = self.model.predict(text_vectorized)[0]
        probabilities = self.model.predict_proba(text_vectorized)[0]
        return {
            "value": prediction,
            "label": cyberbully_labels[prediction],
            "probability": round(probabilities.max() *100, 2)
        }