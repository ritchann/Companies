import re
import csv
import pandas as pd
import nltk


def clean_text(value):
    value = re.sub(r'\([^()]*\)', '', value)
    value = re.sub(r"\d+", "", value, flags=re.UNICODE)

    value = value.lower()

    for ch in ['&', 'corporation', 'group', '*', ',', 'ооо', '"', '/', "'"]:
        value = value.replace(ch, '')

    array = value.split()
    result = []

    for word in array:
        if len(array) == 1:
            result.append(word)
        elif (len(word) > 1) and ('.' not in word):
            result.append(word)

    result_string = ' '.join(result)

    return result_string


def data_transformation():
    train = pd.read_csv("files/train.csv")

    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('corpus')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

    length = len(train)

    rows = []
    header = ['index', 'original', 'transformed']

    for i in range(len(train[1:length])):
        row = train.iloc[i]

        name1 = row['name_1']
        name2 = row['name_2']

        transformed_1 = clean_text(name1)
        transformed_2 = clean_text(name2)

        rows.append([i, name1, transformed_1])
        rows.append([i, name2, transformed_2])

    with open('files/transformed_train.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def get_data():
    #data_transformation()

    transformed_train = pd.read_csv("files/transformed_train.csv")
    print(len(transformed_train))
    print(transformed_train.head())


