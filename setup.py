
from setuptools import setup, find_packages

__version__ = "0.0.1"

with open("requirements.txt", 'r') as f:
    requirements = f.readlines()

# TODO: default back to numpy if cupy can't be installed

setup(
    name="quickernet",
    version=__version__,
    author="Dane Howard",
    author_email="dane.a.howard@gmail.com",
    description="A small, fast neural net experimentation framework",
    url="https://github.com/mirrord/quickernet",
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.6",
    test_suite="unit_tests"
)
