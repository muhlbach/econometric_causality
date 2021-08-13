import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.0.2'
PACKAGE_NAME = 'econometric_causality'
AUTHOR = 'Nicolaj Søndergaard Mühlbach'
AUTHOR_EMAIL = 'n.muhlbach@gmail.com'
URL = 'https://github.com/muhlbach/econometric_causality'

LICENSE = 'MIT License'
DESCRIPTION = 'This repo implements various econommetric and ML approaches to causal inference'
LONG_DESCRIPTION = (HERE / "README.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'numpy>=1.19',
      'pandas>=1.2'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )