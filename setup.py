from setuptools import setup, find_packages

setup(
    name="DecayHeatML",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21',
        'pandas>=1.3',
        'scikit-learn>=1.0',
        'matplotlib>=3.5',
        'ipywidgets>=7.6',
        'gplearn>=0.4',
        'scikit-optimize>=0.9'
    ],
    extras_require={
      "Interactive": ['notebook'],
      "Webapp": ['streamlit']
    },
    python_requires='>=3.8',
)
