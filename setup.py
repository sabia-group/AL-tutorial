from setuptools import setup, find_packages

setup(
    name="alt",  # Your project name
    version="0.1.0",  # Adjust versioning as needed
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "jupytext",
        "torch>=1.12",       # PyTorch is installed via pip as "torch"
        "mace-torch",
    ],
    extras_require={
        "notebooks": ["jupyter"],
    },
    python_requires=">=3.9",
    author="Krystof Brezina, Shubham Sharma, Elia Stocco",
    description="Active Learning Tutorial",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)
