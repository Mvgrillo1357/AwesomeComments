# From the input model collection, calculate and return the ensemble prediction
def get_ensemble_prediction(models, text):
    # Get probability predictions for each model in the input collection
    probabilities = []
    for model in models:
        probabilities.append(model.get_probabilities(text))
    # Sum the corresponding values in each resulting probability array
    probabilities_sum = sum(probabilities)
    # Average each summed probability using the input model collection size
    probabilities_avg = probabilities_sum / len(models)
    # Find the ensemble probability prediction (highest average)
    prediction_probability = probabilities_avg.max()
    # Find the location index of the prediction probability (to get class)
    prediction_index = probabilities_avg.tolist().index(prediction_probability)
    # Get the prediction class using the prediction probability index
    prediction_class = models[0].get_classes()[prediction_index]
    return {
        "class": prediction_class,
        "probability": prediction_probability
    }
