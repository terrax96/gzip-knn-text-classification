from sklearn.datasets import fetch_20newsgroups
from gzip_knn import GZipKNN
from sklearn.metrics import accuracy_score

K = 5
CATS = [
    'comp.os.ms-windows.misc',
    'sci.electronics',
    'soc.religion.christian',
    'rec.sport.baseball',
    'rec.autos',
    'rec.motorcycles'
]


if __name__ == "__main__":
    newsgroups_train = fetch_20newsgroups(subset='train', categories=CATS, data_home="./data/20news")
    newsgroups_test = fetch_20newsgroups(subset='test', categories=CATS, data_home="./data/20news")

    sample = newsgroups_test.data[567]
    label = newsgroups_test.target[567]
    print(sample)
    print("__________________________")

    model = GZipKNN(n_neighbors=K)
    model.fit(newsgroups_train.data, newsgroups_train.target)
    distances, top_k_idx = model.kneighbors(sample)
    
    for i in top_k_idx:
        print(newsgroups_train.data[i])
        print("__________________________")
    
    print(distances, top_k_idx)
    print(newsgroups_train.target[top_k_idx])

    preds = model.predict([sample])
    print("Label:", label, newsgroups_train.target_names[label])
    print("Prediction:", preds[0], newsgroups_train.target_names[preds[0]])

    print("Accuracy:", accuracy_score(newsgroups_test.target, model.predict(newsgroups_test.data)))