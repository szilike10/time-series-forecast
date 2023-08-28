import argparse
import pandas as pd
import re

from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer


def read_excel_sheets(path):
    excel = pd.ExcelFile(path)
    df = pd.DataFrame()

    for sheet_name in excel.sheet_names:
        sheet = excel.parse(sheet_name)
        sheet['furnizor'] = [sheet_name for _ in range(len(sheet))]
        sheet['product_names'] = sheet.agg(
            lambda x: f"{' '.join([s[:4] for s in re.split(r'[ .,]', x['denumire']) if not re.match(r'[0-9].*g', s)])} {x['furnizor'][:4]}".lower(), axis=1)
        # sheet['category'] = sheet.agg(lambda x: f"{' '.join(x['denumire'].split(' ')[:1])}", axis=1)
        sheet['category'] = sheet.agg(lambda x: f"{' '.join(re.split(r'[ .]', x['denumire'])[:1])}".lower(), axis=1)
        sheet['um'] = sheet['um'].fillna('buc')
        sheet['um'] = sheet['um'].map(lambda x: x.lower())
        df = pd.concat([df, sheet])

    df = df.reset_index()
    df = df.drop('index', axis=1)

    for i, e in enumerate(df['category']):
        if e in ['carnati', 'ceafa', 'cotlet', 'piept', 'salam', 'slanina', 'sunca']:
            df.at[i, 'category'] = 'carne'
        elif e in ['br', 'branza']:
            df.at[i, 'category'] = 'branza'

    clusters, n_clusters = cluster_items(df['product_names'])
    df['cluster'] = clusters

    return df


def save_cmobined_csv(df, out_path):
    df.to_csv(out_path, index=False)


def cluster_items(x):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(x)

    dbscan = DBSCAN(eps=1, min_samples=5, metric='euclidean')
    dbscan.fit(tfidf_matrix)

    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    return labels, n_clusters


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel_input', type=str)
    parser.add_argument('--out_csv_path', type=str)
    args = parser.parse_args()

    path_to_excel = args.excel_input
    out_csv_path = args.out_csv_path

    if path_to_excel is None or out_csv_path is None:
        raise Exception('Please specify the input excel sheet path and the csv output path.')

    df = read_excel_sheets(path_to_excel)
    save_cmobined_csv(df, out_csv_path)
