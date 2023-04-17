"""Setup file."""

from setuptools import setup

with open("README.md") as readme_file:
    readme = readme_file.read()

requirements = [
    "matplotlib",
    "numpy",
    "pandas",
    "tensorflow",
    "tensorflow-probability",
    "jax",
]

setup(
    name="lcmcmc",
    version="0.0.1",
    author="Biswajit Biswas",
    author_email="biswas@apc.in2p3.fr",
    maintainer="Biswajit Biswas",
    maintainer_email="biswajit.biswas@apc.in2p3.fr",
    description="Fitting lightcurves wiht MC sampling",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/b-biswas/MADNESS",
    include_package_data=True,
    packages=["lcmcmc"],
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    package_data={"lcmcmc": ["data/*"]},
)