# From a provided model collection, calculate and return the ensemble prediction
def get_ensemble_prediction(models, text):
    # Get prediction probabilities for each model in the collection
    probabilities = []
    for model in models:
        probabilities.append(model.get_probabilities(text))
    # Sum the corresponding values in each resulting probability array
    probabilities_sum = sum(probabilities)
    # Average each summed probability using the collection size
    probabilities_avg = probabilities_sum / len(models)
    # Find the ensemble prediction probability (highest average)
    prediction_probability = probabilities_avg.max()
    # Find the location index of the prediction probability (to get the class next)
    prediction_index = probabilities_avg.tolist().index(prediction_probability)
    # Get the prediction class using the prediction index
    prediction_class = models[0].get_classes()[prediction_index]
    return {
        "class": prediction_class,
        "probability": prediction_probability
    }
