import gzip
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from typing import Iterable

# https://aclanthology.org/2023.findings-acl.426.pdf

K = 5
CATS = [
    'comp.os.ms-windows.misc',
    'sci.electronics',
    'soc.religion.christian',
    'rec.sport.baseball',
    'rec.autos',
    'rec.motorcycles'
]

def predict(X_train, y_train, x1: str):
    Cx1 = len(gzip.compress(x1.encode()))
    distance_from_x1 = []

    for x2 in X_train:
        Cx2 = len(gzip.compress(x2.encode()))
        x1x2 = " ".join([x1, x2])
        Cx1x2 = len(gzip.compress(x1x2.encode()))
        ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2) # Normalized Compression Distance
        distance_from_x1.append(ncd)

    sorted_idx = np.argsort(np.array(distance_from_x1))
    print(sorted_idx[:K])
    top_k_classes = y_train[sorted_idx[:K]].tolist()
    print(top_k_classes)
    predict_class = max(set(top_k_classes), key=top_k_classes.count)
    return predict_class


def test(training_set, test_set):
    predictions = []
    for x1, _ in test_set:
        Cx1 = len(gzip.compress(x1.encode()))
        distance_from_x1 = []

        for x2, _ in training_set:
            Cx2 = len(gzip.compress(x2.encode()))
            x1x2 = " ".join([x1, x2])
            Cx1x2 = len(gzip.compress(x1x2.encode()))
            ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2) # Normalized Compression Distance
            distance_from_x1.append(ncd)

        sorted_idx = np.argsort(np.array(distance_from_x1))
        top_k_classes = training_set[sorted_idx[:K], 1]
        print(top_k_classes)
        predict_class = max(set(top_k_classes), key=top_k_classes.count)
        predictions.append(predict_class)


class GZipKNN:
    def __init__(self, n_neighbors) -> None:
        self.n_neighbors = n_neighbors

    def fit(self, X: Iterable[str], y: Iterable[int]):
        self.X = X
        self.y = y
    
    def predict(self, X: Iterable[str]):
        predictions = []
        for x1 in X:
            Cx1 = len(gzip.compress(x1.encode()))
            distance_from_x1 = []

            for x2 in self.X:
                Cx2 = len(gzip.compress(x2.encode()))
                x1x2 = " ".join([x1, x2])
                Cx1x2 = len(gzip.compress(x1x2.encode()))
                ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2) # Normalized Compression Distance
                distance_from_x1.append(ncd)

            sorted_idx = np.argsort(np.array(distance_from_x1))
            top_k_classes = self.y[sorted_idx[:self.n_neighbors]].tolist()
            predict_class = max(set(top_k_classes), key=top_k_classes.count)
            predictions.append(predict_class)
        return predictions



if __name__ == "__main__":
    newsgroups_train = fetch_20newsgroups(subset='train', categories=CATS, data_home="./data/20news")
    newsgroups_test = fetch_20newsgroups(subset='test', categories=CATS, data_home="./data/20news")

    # class_id = predict(newsgroups_train.data, newsgroups_train.target, x1=newsgroups_test.data[0])
    # print(newsgroups_test.data[0])
    # print(class_id, newsgroups_train.target_names[class_id])

    # for i in [357, 1258, 2717, 2790, 1815]: # [0, 2, 0, 0, 4]
    #     print(newsgroups_train.data[i])
    #     print("_________________________")

    model = GZipKNN(n_neighbors=K)
    model.fit(newsgroups_train.data, newsgroups_train.target)
    preds = model.predict([newsgroups_test.data[0]])
    print(preds, [newsgroups_train.target_names[pred] for pred in preds])
