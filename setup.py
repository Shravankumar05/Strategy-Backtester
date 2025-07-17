from setuptools import setup, find_packages

setup(
    name="strategy-backtester",
    version="0.1.0",
    description="A trading strategy backtester built with Streamlit",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "yfinance>=0.2.0",
        "streamlit>=1.30.0",
        "pytest>=7.0.0",
        "pydantic>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "great-expectations>=0.17.0",
    ],
    python_requires=">=3.8",
)