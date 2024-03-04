from sklearn.datasets import fetch_20newsgroups
from gzip_knn import GZipKNN

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
