from typing import Iterable
import gzip
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

class GZipKNN:
    def __init__(self, n_neighbors: int, n_processes: int | None = None) -> None:
        self.n_neighbors = n_neighbors
        self.n_processes = cpu_count() if n_processes is None else n_processes

    def fit(self, X: Iterable[str], y: Iterable[int]) -> None:
        self.X = X
        self.y = y

    def kneighbors(self, x1: str, n_neighbors: int | None = None, return_distance: bool = True) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        if n_neighbors is None:
            n_neighbors = self.n_neighbors
        Cx1 = len(gzip.compress(x1.encode()))
        distance_from_x1 = []

        for x2 in self.X:
            Cx2 = len(gzip.compress(x2.encode()))
            x1x2 = " ".join([x1, x2])
            Cx1x2 = len(gzip.compress(x1x2.encode()))
            ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2) # Normalized Compression Distance
            distance_from_x1.append(ncd)

        distance_from_x1 = np.array(distance_from_x1)
        sorted_idx = np.argsort(distance_from_x1)
        top_k_idx = sorted_idx[:n_neighbors]
        if return_distance:
            return distance_from_x1[top_k_idx], top_k_idx
        else:
            return top_k_idx
        
    def _predict(self, x1: str) -> int:
        top_k_idx = self.kneighbors(x1, return_distance=False)
        top_k_classes = self.y[top_k_idx].tolist()
        predict_class = max(set(top_k_classes), key=top_k_classes.count)
        return predict_class
    
    def predict(self, X: Iterable[str]) -> np.ndarray:
        with Pool(self.n_processes) as pool:
            predictions = list(tqdm(pool.imap(self._predict, X), total=len(X)))
        return np.array(predictions)