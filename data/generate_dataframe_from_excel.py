import argparse
import pandas as pd


def read_excel_sheets(path):
    excel = pd.ExcelFile(path)
    df = pd.DataFrame()

    for sheet_name in excel.sheet_names:
        sheet = excel.parse(sheet_name)
        sheet['furnizor'] = [sheet_name for _ in range(len(sheet))]
        df = pd.concat([df, sheet])

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
