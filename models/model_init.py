import os
import pickle
# from keras.preprocessing.text import Tokenizer
# from tensorflow import keras
# from keras.preprocessing.sequence import pad_sequences

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


# # SENTIMENT
# POSITIVE = "POSITIVE"
# NEGATIVE = "NEGATIVE"
# NEUTRAL = "NEUTRAL"
# SENTIMENT_THRESHOLDS = (0.4, 0.7)

# class SequentialModel:
#     def __init__(self, model_name):
#         self.model = keras.models.load_model(os.path.join(os.path.dirname(__file__), 'models', f'sentiment_{model_name}.h5'))

#     def get_sequential_sentiment(self, text):
#         tokenizer = Tokenizer()
#         # Tokenize text
#         x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=300)

#         prediction = self.model.predict([x_test])[0]
#         return {
#             "value": prediction,
#             "label": sentiment_labels[0],
#             "probability": round(prediction.max() * 100, 2)
#         }
