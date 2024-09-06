import setuptools

setuptools.setup(
    name="Fake-Real-News-Classification",
    packages=setuptools.find_packages(),
    install_requires = [
        "datasets",
        "gdown",
        "tqdm",
        "numpy",
        "ipywidgets",
        "pyparsing",
    ],
)
