import os
from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="PhD Project",
    version="0.0.1",
    author="D'Jeff, Nkashama Kanda",
    author_email="",
    description="This is the ML project as part of my PhD work on anomaly detection for"
                "industrial control system",
    license="BSD",
    keywords="Pytorch deep learning cnn active learning",
    url="",
    packages=find_packages(exclude=['contrib', 'doc', 'unit_tests']),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
)
