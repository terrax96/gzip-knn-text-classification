from typing import Iterable
import gzip
import numpy as np

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