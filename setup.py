import os
import sys
import shutil
import warnings

from io import open
from pathlib import Path
from setuptools import setup, find_packages
from setuptools.command.install import install

here = Path(__file__).parent.resolve()

sys.path.insert(0, str(here / 'jax_meta'))
from version import VERSION

class InstallCommand(install):
    def run(self):
        super().run()

        if 'JAX_META_REPOSITORY' in os.environ:
            repository = Path(os.environ['JAX_META_REPOSITORY'])
            if not repository.exists():
                warnings.warn(f'Repository not found (ignoring): {repository}')
                return

            destination = Path(self.install_lib) / 'jax_meta' / 'datasets' / 'assets'
            shutil.copy(repository, destination / 'repository.json')

# Get the long description from the README file
with open(here / 'README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='jax_meta',
    version=VERSION,
    description='Collection of meta-learning algorithms in Jax',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    author='Tristan Deleu',
    author_email='tristan.deleu@gmail.com',
    url='https://github.com/tristandeleu/jax-meta-learning',
    keywords=['meta-learning', 'jax', 'few-shot', 'few-shot learning'],
    packages=find_packages(exclude=['data', 'contrib', 'docs', 'tests', 'examples']),
    install_requires=[
        'jax',
        'jaxlib',
        'tqdm',
        'simple-parsing',
        'Pillow>=7.0.0',
        'dm-haiku',
        'optax'
    ],
    package_data={'jax_meta': ['jax_meta/datasets/assets/*']},
    include_package_data=True,
    cmdclass={'install': InstallCommand},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'License :: OSI Approved :: MIT License',
    ],
)
