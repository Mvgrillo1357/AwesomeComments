def get_ensemble_prediction(models, text):
    # Get probability predictions for each model
    probabilities = []
    for model in models:
        probabilities.append(model.get_probabilities(text))
    # Sum the corresponding values in each probability array
    probabilities_sum = sum(probabilities)
    # Average each summed probability using the model count
    probabilities_avg = probabilities_sum / len(models)
    # Find the prediction (highest) probability
    prediction_probability = probabilities_avg.max()
    # Find the location index of the prediction
    prediction_index = probabilities_avg.tolist().index(prediction_probability)
    # Get the prediction class using the location index
    prediction_class = models[0].get_classes()[prediction_index]
    return {
        "class": prediction_class,
        "probability": round(prediction_probability * 100, 2)
    }
