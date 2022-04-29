import os
import pickle


# Display label dictionary for sentiment classes
sentiment_labels = {
    0: "Negative",
    4: "Positive",
    # Some models are transforming 4 to 1
    1: "Positive"
}

# Display label dictionary for cyberbullying classes
cyberbullying_labels = {
    "age": "Ageism",
    "ethnicity": "Racism",
    "gender": "Sexism",
    "not_cyberbullying": "Not Cyberbullying",
    "other_cyberbullying": "Another Type of Cyberbullying",
    "religion": "Anti Religious"
}


# Model class for common interface on serialized models and input vectorizers
class Model:
    # Initializes the Model instance
    # files are assumed to be in the same directory as this file (/models)
    def __init__(self, model_file: str, vectorizer_file: str):
        # Build file paths using current system directory
        model_path = os.path.join(os.path.dirname(__file__), model_file)
        vectorizer_path = os.path.join(os.path.dirname(__file__), vectorizer_file)
        # Unpickle the files in read mode and set as instance variables
        self.model = pickle.load(open(model_path, 'rb'))
        self.vectorizer = pickle.load(open(vectorizer_path, 'rb'))

    def get_classes(self):
        # Returns the classes from the instance's model
        return self.model.classes_

    def get_probabilities(self, text: str):
        # Vectorize the input text using the instance's vectorizer
        text_vectorized = self.vectorizer.transform([text])
        # Return the instance model's precitions for the input text
        return self.model.predict_proba(text_vectorized)[0]
