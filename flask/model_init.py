import pickle

sentiment_labels = {
    0: "Negative",
    1: "Positive",
    4: "Positive"
}

cyberbully_labels = {
    "age" : "Ageism",
    "ethnicity" : "Racism",
    "gender" : "Ageism",
    "not_cyberbullying" : "Not Cyberbullying",
    "other_cyberbullying" : "Another Type of Cyberbullying",
    "religion" : "Anti Religious"
}

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