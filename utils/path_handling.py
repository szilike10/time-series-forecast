from pathlib import Path


def handle_parent_path(filepath):
    sep = '/'
    if '\\' in filepath:
        sep = '\\'
    dir_path = Path(filepath.rsplit('/', 1)[0])
    dir_path.mkdir(parents=True, exist_ok=True)
