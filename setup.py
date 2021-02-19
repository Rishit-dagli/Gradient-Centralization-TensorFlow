from setuptools import setup

exec(open('gctf/version.py').read())

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="gradient-centralization-tf",
    version="0.0.1",
    description="Implement Gradient Centralization in TensorFlow",
    py_modules=["gctf"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: Apache Software License",
    ],
    url="https://github.com/Rishit-dagli/Gradient-Centralization-TensorFlow",
    author="Rishit Dagli",
    author_email="rishit.dagli@gmail.com",
    install_requires=[
        "tensorflow ~= 2.4.0",
        "keras ~= 2.4.0",
    ],
    extras_require={
        "dev": [
            "check-manifest",
            "twine",
            "numpy"
        ],
    },
)
