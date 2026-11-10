# 1.) Potřebné importy knihoven
import os
import re
import nltk
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize

# 2.) Cesty k souborům (relativní cesty)
base_path = Path(__file__).parent
path_to_docs = base_path / "documents"
path_to_queries = base_path / "query_devel.xml"
output_filename = base_path / "vystup.txt"

# 3.) Načtení dokumentů
documents_raw = {}
for document_name in os.listdir(path_to_docs):
    with open(os.path.join(path_to_docs, document_name), 'r', encoding='utf-8') as document:
        documents_raw[document_name] = document.read()
print(f"Načteno {len(documents_raw)} dokumentů.") 
# print("První dokument:")
# print(list(documents_raw.items())[0])

# 4.) Předzpracování dokumentů
documents_cleaned = {}
for doc_name, content in documents_raw.items():
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped_line = ''.join(line.split())
        if stripped_line.replace('.', '', 1).isdigit():
            continue
        if line in ['<html>', '</html>', '<pre>', '</pre>', '']:
            continue
        cleaned_lines.append(line)
    documents_cleaned[doc_name] = ' '.join(cleaned_lines)
# print(f"Dokumenty předzpracovány. První vyčištěný dokument:")
# print(list(documents_cleaned.items())[0])

# 5.) Načtení dotazů
queries_raw = []
with open(path_to_queries, 'r', encoding='utf-8') as file:
    for line in file:
        if line.strip():
            queries_raw.append(line.strip())
# print(f"Načteno {len(queries_raw)} řádků dotazů. Prvních 5 řádků:")
# print(queries_raw[:5])

# 6.) Předzpracování dotazů
doc_start_indices = []
doc_end_indices = []
for idx, line in enumerate(queries_raw):
    if line.strip() == '<DOC>':
        doc_start_indices.append(idx)
    elif line.strip() in ['</DOC>', '</DOC> ']:
        doc_end_indices.append(idx)

queries_cleaned = {}
for i in range(len(doc_start_indices)):
    topic_line = queries_raw[doc_start_indices[i] + 1]
    match = re.search(r'\d+', topic_line)
    if match:
        topic_id = int(match.group())
    else:
        print(f"Nepodařilo se získat číslo tématu v řádku: {topic_line}")
        continue

    extracted_text = ' '.join(queries_raw[doc_start_indices[i] + 2:doc_end_indices[i]])
    queries_cleaned[topic_id] = extracted_text
# print(f"Vyčištěno {len(queries_cleaned)} dotazů. První dotaz:")
# print(list(queries_cleaned.items())[0])

# 7.) Stemming - redukování slov na kořen
stemmer = nltk.PorterStemmer()
for key in documents_cleaned:
    words = word_tokenize(documents_cleaned[key])
    documents_cleaned[key] = ' '.join(stemmer.stem(word) for word in words)
for key in queries_cleaned:
    words = word_tokenize(queries_cleaned[key])
    queries_cleaned[key] = ' '.join(stemmer.stem(word) for word in words)
# print("Stemming proveden. První stemmovaný dokument:")
# print(list(documents_cleaned.items())[0])
# print("Stemming proveden #2. První stemmovaný dotaz:")
# print(list(queries_cleaned.items())[0])

# 8.) Tf-idf vektorizace a výpočet kosinové podobnosti
data_list = list(documents_cleaned.values())
document_names = [os.path.splitext(name)[0] for name in documents_cleaned.keys()]
tfidf = TfidfVectorizer(
    max_features= 10000,
    use_idf=True, 
    smooth_idf=True, 
    sublinear_tf=True, 
    stop_words='english'
)
sparse_doc_term_matrix = tfidf.fit_transform(data_list)
# print(f"TF-IDF matice vytvořena. Tvar matice: {sparse_doc_term_matrix.shape}")

# 9.) Tvorba výstupního souboru
with open(output_filename, 'w') as f:
    for topic_id, query in queries_cleaned.items():
        query_vector = tfidf.transform([query])
        similarities = cosine_similarity(sparse_doc_term_matrix, query_vector).flatten()
        sorted_indices = np.argsort(similarities)[::-1]
        for rank in range(min(100, len(sorted_indices))):
            f.write(f"{topic_id}\t{document_names[sorted_indices[rank]]}\t{similarities[sorted_indices[rank]]}\n")
print(f"Výstup byl uložen do souboru: {output_filename}")