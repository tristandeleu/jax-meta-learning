import os
import json
import shutil

from urllib.request import urlretrieve
from pathlib import Path


def download_url(url, root, filename=None):
    root = root.expanduser()
    if filename is None:
        filename = os.path.basename(url)
    filepath = root / filename

    if filepath.is_file():
        return

    root.mkdir(exist_ok=True)
    urlretrieve(url, filename=filepath)
    return filepath


def get_asset_path(*args):
    jax_meta_folder = Path(__file__).parent.parent
    assets_folder = jax_meta_folder / 'datasets' / 'assets'
    return assets_folder.joinpath(*args)


def get_asset(*args, dtype=None):
    filename = get_asset_path(*args)
    if not filename.exists():
        raise IOError(f'File not found: {filename}')

    if dtype is None:
        dtype = filename.suffix
        dtype = dtype[1:]

    if dtype == 'json':
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        raise NotImplementedError(f'Unknown type: {dtype}')
    return data


def copy_dataset_from_repository(name, destination):
    try:
        repository = get_asset('repository.json', dtype='json')
    except IOError:
        raise NotImplementedError(f'Using the dataset `{name}` in jax_meta '
            'requires a private repository of datasets, and therefore is not '
            'publicly available yet. This will be made available in a future release.')

    filename = Path(repository['datasets'][name]['local'])
    shutil.copy(filename, destination / filename.name)
    return destination / filename.name


def update_repository(origin=None):
    repo_filepath = get_asset_path('repository.json')
    if origin is None:
        repository = get_asset('repository.json', dtype='json')
        origin = Path(repository['meta']['origin'])
    shutil.copy(origin, repo_filepath)
