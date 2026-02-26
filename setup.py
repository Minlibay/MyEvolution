from setuptools import setup, find_packages

setup(
    name="evolution-simulation",
    version="0.1.0",
    description="Agent-based simulation of tool evolution",
    author="Research Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "dev": ["pytest>=6.2.0", "jupyter>=1.0.0"],
        "viz": ["plotly>=5.0.0", "dash>=2.0.0"],
    },
    python_requires=">=3.8",
)
