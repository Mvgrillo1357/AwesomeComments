import os
import pickle

sentiment_labels = {
    0: "Negative",
    1: "Positive"
}

cyberbullying_labels = {
    "age": "Ageism",
    "ethnicity": "Racism",
    "gender": "Sexism",
    "not_cyberbullying": "Not Cyberbullying",
    "other_cyberbullying": "Another Type of Cyberbullying",
    "religion": "Anti Religious"
}

class Model:
    def __init__(self, model_file: str, vectorizer_file: str):
        model_path = os.path.join(os.path.dirname(__file__), model_file)
        self.model = pickle.load(open(model_path, 'rb'))
        vectorizer_path = os.path.join(os.path.dirname(__file__), vectorizer_file)
        self.vectorizer = pickle.load(open(vectorizer_path, 'rb'))

    def get_classes(self):
        return self.model.classes_

    def get_probabilities(self, text: str):
        text_vectorized = self.vectorizer.transform([text])
        probabilities = self.model.predict_proba(text_vectorized)[0]
        return probabilities
