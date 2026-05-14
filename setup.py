from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="black-scholes-model",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A complete Python implementation of the Black-Scholes options pricing model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/black-scholes-model",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "jupyter>=1.0.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
    ],
    keywords="black-scholes options pricing quantitative-finance greeks implied-volatility",
)
