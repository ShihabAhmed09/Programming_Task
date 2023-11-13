import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


def create_data():
    normal = np.random.uniform(0, 10, size=(100, 1000))
    normal.tofile("normal.bin")

    abnormal = np.random.uniform(5, 15, size=(100, 1000))
    abnormal.tofile("abnormal.bin")


def load_data():
    normal = np.fromfile("normal.bin").reshape(100, 1000)
    abnormal = np.fromfile("abnormal.bin").reshape(100, 1000)

    return normal, abnormal


def split_data(normal, abnormal):
    training, normal_test = normal[:90], normal[90:]
    abnormal_test = abnormal[:10]

    test = np.vstack((normal_test, abnormal_test))

    return training, test


def calculate_dissimilarity_scores(training):
    baseline = []
    for i in range(len(training)):
        distances = euclidean_distances(training[i].reshape(1, -1), training)
        sorted_distances = np.sort(distances)
        top_distances = sorted_distances[:, 1:6]
        score = top_distances.sum()
        baseline.append(score)
    return baseline


def print_predictions(training, test, baseline):
    test_scores = []
    for i in range(len(test)):
        distances = euclidean_distances(test[i].reshape(1, -1), training)
        sorted_distances = np.sort(distances)
        top_distances = sorted_distances[:, 1:6]
        score = top_distances.sum()
        test_scores.append(score)

    min_baseline_value = min(baseline)
    max_baseline_value = max(baseline)

    predictions = []
    for score in test_scores:
        if min_baseline_value <= score <= max_baseline_value:
            predictions.append("Normal")
        else:
            predictions.append("Abnormal")

    return predictions


if __name__ == "__main__":
    create_data()
    normal, abnormal = load_data()
    training, test = split_data(normal, abnormal)
    baseline = calculate_dissimilarity_scores(training)
    predictions = print_predictions(training, test, baseline)

    for i, prediction in enumerate(predictions):
        print(f"Data {i}: {prediction}")
