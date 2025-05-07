from setuptools import setup, find_packages

setup(
    name="autolrc",
    version="1.0.3",
    packages=find_packages(),
    install_requires=[
        "pydub",
        "google-generativeai",
    ],
)