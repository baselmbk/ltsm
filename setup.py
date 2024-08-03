from setuptools import setup, find_packages

setup(
    name="crypto_prediction",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "tensorflow",
        "pymongo",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "ta-lib",
    ],
    extras_require={
        "dev": [
            "pytest",
            "flake8",
        ],
    },
    entry_points={
        "console_scripts": [
            "crypto_predict=src.main:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A cryptocurrency price prediction system using LSTM and ensemble models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/crypto_prediction",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)