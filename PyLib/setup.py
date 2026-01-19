from setuptools import setup
from distutils.command.build_py import build_py

with open('README.md', 'r') as f:
    long_description = f.read()

from src import (
    __version__, 
    __author__,
    __email__,
    __description__,
    __status__
)

setup(
    name='hepherolib',
    version=__version__,
    description=__description__,
    long_description=long_description,
    url='https://github.com/DESY-CBPF-UERJ/HEPHeroLib',
    license='MIT License',
    author=__version__,
    author_email=__email__,
    packages=[
    	'hepherolib',
    	'hepherolib.analysis',
    	'hepherolib.data',
        'hepherolib.btageffmap'
    ],
    package_dir={
    	'hepherolib': 'src',
    	'hepherolib.analysis': 'src/analysis',
    	'hepherolib.data': 'src/data',
    	'hepherolib.btageffmap': 'src/btageffmap'
    },
    install_requires=[
        "awkward0>=0.15.5",
        "cachetools>=4.2.1",
        "cycler>=0.10.0",
        "kiwisolver>=1.3.1",
        "matplotlib>=3.4.1",
        "mplhep>=0.3.2",
        "mplhep-data>=0.0.2",
        "numpy>=1.19.2",
        "packaging>=20.9",
        "pandas>=1.2.3",
        "Pillow>=8.2.0",
        "pyparsing>=2.4.7",
        "python-dateutil>=2.8.1",
        "pytz>=2021.1",
        "six>=1.15.0",
        "uhi>=0.2.1",
        "uproot3>=3.14.4",
        "uproot3-methods>=0.10.1",
        "tqdm>=4.59.0",
        #"futures>=3.1.1",
        "scikit-learn>=0.24.2",
        "statsmodels>=0.12.2",
        "h5py>=2.10.0",
        "iminuit>=2.25.2",
        "onnxruntime>=1.19.2",
    ],
    cmdclass={
        'build_py': build_py
    },
    classifiers=[
        f"Development Status :: 5 - {__status__}",
        "Programming Language :: Python",
        "License :: Private",
    ],
)

with open("./install.sh", "w") as f:
    f.write(f"pip3 install ./dist/hepherolib-{__version__}.tar.gz")
