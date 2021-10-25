from setuptools import find_packages, setup

setup(
    name="fantasticfeatures",
    version="0.1",
    long_description=__doc__,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        "statsmodels==0.10.2",
        "matplotlib==3.2.2",
        "sklearn>=0.0",
        "sklearn-pandas==1.8.0",
        "scikit-learn==0.22.2.post1",
    ],
)
