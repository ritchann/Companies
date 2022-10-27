import pickle
import numpy as np
import pandas as pd
from dataset import clean_text
import fastwer
import scipy.linalg.decomp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples
from gensim.models import Word2Vec
from nltk import word_tokenize


def vectorize(list_of_docs, model):
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features


def mbkmeans_clusters(X, k, batch_size, print_silhouette_values):
    km = MiniBatchKMeans(n_clusters=k, batch_size=batch_size).fit(X)

    if print_silhouette_values:
        sample_silhouette_values = silhouette_samples(X, km.labels_)
        print(km.labels_)
        silhouette_values = []
        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
            silhouette_values.append(
                (
                    i,
                    cluster_silhouette_values.shape[0],
                    cluster_silhouette_values.mean(),
                    cluster_silhouette_values.min(),
                    cluster_silhouette_values.max(),
                )
            )

    return km, km.labels_


def similar_names_word2_vec(name):
    df = pd.read_csv("files/tokens_data.csv")
    clustering = pickle.load(open("files/clusteringW2Vec.pkl", "rb"))
    tokenized_docs = df["tokens"].values
    docs = df["original"].values

    model = Word2Vec(sentences=tokenized_docs, workers=1, seed=1988)
    vectorized_docs = vectorize(tokenized_docs, model=model)
    x = vectorize([name], model=model)

    test_cluster = clustering.predict(x)
    most_representative_docs = np.argsort(
        np.linalg.norm(vectorized_docs - clustering.cluster_centers_[test_cluster], axis=1))

    print("Similar names for %s:" % name)

    for d in most_representative_docs:
        cer = fastwer.score_sent(clean_text(docs[d]), clean_text(name), char_level=True)

        if cer < 40:
            print(docs[d])
            print("------------------------------")


def word2_vec():
    text_columns = ["original", "transformed"]
    df_raw = pd.read_csv("files/transformed_train.csv")
    df = df_raw.copy()

    for col in text_columns:
        df[col] = df[col].astype(str)

    df["tokens"] = df["transformed"].map(lambda x: word_tokenize(x))

    _, idx = np.unique(df["tokens"], return_index=True)
    df = df.iloc[idx, :]

    tokenized_docs = df["tokens"].values

    model = Word2Vec(sentences=tokenized_docs, workers=1, seed=1988)

    vectorized_docs = vectorize(tokenized_docs, model=model)

    clustering, cluster_labels = mbkmeans_clusters(X=vectorized_docs, k=50, batch_size=500,
                                                   print_silhouette_values=True)
    pickle.dump(clustering, open("files/clusteringW2Vec.pkl", "wb"))


def similar_names_tf_idf(name):
    model = pickle.load(open("files/modelTfIdf.pkl", "rb"))
    td_df = pd.read_csv("files/tf_idf_data.csv")
    transformed_train = pd.read_csv("files/transformed_train.csv")
    transformed_train['transformed'] = transformed_train['transformed'].astype(str)
    search = clean_text(name)

    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit_transform(transformed_train.transformed.to_list())
    y = vectorizer.transform([search])
    org_prediction = model.predict(y)
    length = len(td_df)

    print("Similar names for %s:" % name)

    for i in range(len(td_df[0:length])):
        row = td_df.iloc[i]
        text = row.transformed

        if row.cluster_tf_idf == str(org_prediction):
            cer = fastwer.score_sent(search, text, char_level=True)

            if cer < 40:
                print(row.original)
                print("------------------------------")


def tf_idf():
    true_k = 200
    transformed_train = pd.read_csv("files/transformed_train.csv")

    vectorizer = TfidfVectorizer(stop_words='english')
    x = vectorizer.fit_transform(transformed_train.transformed.to_list())

    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(x)

    pickle.dump(model, open("files/modelTfIdf.pkl", "wb"))


def is_duplicate_levenshtein_distance(name_1, name_2):
    cer = fastwer.score_sent(clean_text(name_1), clean_text(name_2), char_level=True)

    if cer > 10:
        print("%s and %s duplicates." % (name_1, name_2))
    else:
        print("%s and %s are not duplicates." % (name_1, name_2))


def levenshtein_distance():
    train = pd.read_csv("files/train.csv")
    sum_true = 0
    length = len(train)

    for i in range(len(train[0:length])):
        row = train.iloc[i]

        name_1 = clean_text(row.name_1)
        name_2 = clean_text(row.name_2)

        cer = fastwer.score_sent(name_1, name_2, char_level=True)

        if cer == scipy.linalg.decomp.inf:
            cer = 0

        is_duplicate = True

        if cer > 10:
            is_duplicate = False

        if int(is_duplicate) == row.is_duplicate:
            sum_true += 1

    print("Accuracy for Levenshtein distance is: ", sum_true / len(train[0:length]))
