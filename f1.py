import pandas as pd
from sklearn import metrics

# Script that outputs confusion matrix and F1 measures (in different variations) from ground truth and prediction file

categories = ["deleted"]

def main():
    # Data must contain column 'deleted' with values between 0,1
    truth = pd.read_csv("truth.csv")
    prediction = pd.read_csv("prediction.csv")

    get_f1_results(truth, prediction, "Prediction")

def to_binary(predictions):
    for category in categories:
        predictions[category] = [1 if row > 0.5 else 0 for row in predictions[category]]
    return predictions


def get_f1_results(truth, predictions, name):

    predictions = to_binary(predictions)

    print(name + ": " + str(metrics.classification_report(truth[categories], predictions[categories])))
    print("Micro: " + str(metrics.f1_score(truth[categories], predictions[categories], average='micro')))
    print("Macro: " + str(metrics.f1_score(truth[categories], predictions[categories], average='macro')))
    print("Average: " + str(metrics.f1_score(truth[categories], predictions[categories], average='weighted')))


if __name__ == "__main__":
    main()
