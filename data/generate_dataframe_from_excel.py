import argparse
import pandas as pd
import re


def read_excel_sheets(path):
    excel = pd.ExcelFile(path)
    df = pd.DataFrame()

    for sheet_name in excel.sheet_names:
        sheet = excel.parse(sheet_name)
        sheet['furnizor'] = [sheet_name for _ in range(len(sheet))]
        # sheet['category'] = sheet.agg(lambda x: f"{' '.join(x['denumire'].split(' ')[:1])}", axis=1)
        sheet['category'] = sheet.agg(lambda x: f"{' '.join(re.split(r'[ .]', x['denumire'])[:1])}".lower(), axis=1)
        df = pd.concat([df, sheet])

    df = df.reset_index()
    df = df.drop('index', axis=1)

    for i, e in enumerate(df['category']):
        if e in ['carnati', 'ceafa', 'cotlet', 'piept', 'salam', 'slanina', 'sunca']:
            df.at[i, 'category'] = 'carne'
        elif e in ['br', 'branza']:
            df.at[i, 'category'] = 'branza'
    return df


def save_cmobined_csv(df, out_path):
    df.to_csv(out_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel_input', type=str)
    parser.add_argument('--out_csv_path', type=str)
    args = parser.parse_args()

    path_to_excel = args.excel_input
    out_csv_path = args.out_csv_path

    df = read_excel_sheets(path_to_excel)
    save_cmobined_csv(df, out_csv_path)
