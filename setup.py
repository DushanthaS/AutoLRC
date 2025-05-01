from setuptools import setup, find_packages

setup(
    name="autolrc",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydub",
        "google-generativeai",
    ],
)