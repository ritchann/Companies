# Company names

## Task
Comparison of company names (they are duplicates or not) and search for similar companies in the database.

<br/>

## Dataset
[Link to the dataset.](https://drive.google.com/file/d/1e9bdr7wcQX_YBudQcsKj-sMoIGxQOlK4/view)

pair_id - pair number

name_1 - name of the first company

name_2 - name of the second company

is_duplicate - are the companies the same

<br/>

Example:
| pair_id | name_1 | name_2 |  is_duplicate |
|----------------|:---------:|----------------:|----------------:|
| 1 | JX Nippon Oil & Gas Exploration (Brasil) Ltda | JX Nippon Oil & Gas Exploration Technical Services Corporation | 1 |
| 2 | Basf Turk Kimya San.Ve Tic Ltd.Sti | Kia Inc. | 0 |

<br/>

## Preprocessing

- removing brackets () and internal text
- removing numbers
- removing punctuation marks
- removing the words 'corporation' and 'group'
- lowercase conversion
- removing short abbreviations that end with a dot

<br/>


| original | transformed | 
|----------------|:---------:|
| Logistics Solutions Ltd. | logistics solutions | 
| Anzco Foods (Europe) Ltd. | anzco foods | 

[Transformed dataset.](https://github.com/ritchann/Companies/blob/main/files/transformed_train.csv)

<br/>

## Models

### The Levenshtein distance is used to compare two company names.

The accuracy  measured using the Levenshtein distance with Character Error Rate (CER > 10) - 0.98

<br/>

### Search for similar names
1. TF-IDF, K-means clustering, Levenshtein distance

2. Word2vec, MiniBatchKMeans, Levenshtein distance

<br/>

## Performance :white_check_mark: Это уже сделано    

CPU: Intel i5-10210U CPU @ 1.60GHz


To compare two values using the Levenshtein distance: 6550/1sec

Speed of processing a request for similar names(Word2vec, MiniBatchKMeans, Levenshtein distance): 0.47sec

<br/>

## Usage

You can open tutorial.ipynb to demonstrate the work of the project. Before using it, you need to install the project dependencies:


```
pip install -r requirements.txt 
```

You can also test the project using the terminal.
<br/>

Levenshtein distance:
```
python train.py --m ld --name1 "Name 1" --name2 "Name 1" 
```
<br/>

TF-IDF, K-means clustering, Levenshtein distance:
```
python train.py --m tf --name1 "Name" 
```
<br/>

Word2vec, MiniBatchKMeans, Levenshtein distance:
```
python train.py --m w2 --name1 "Name"
```

