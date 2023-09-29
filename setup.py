from setuptools import setup, find_packages

setup(
    name='signature_mahalanobis_knn',  # Replace with your desired package name
    version='0.1.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        # Add other dependencies here
    ],
    package_data={
        'test.data': ['*'],
    },
    dependency_links=['https://github.com/sz85512678/sktime'
    ],
    author='Zhen Shao',
    description='Package to detect anomalous streams, or using it to compute the variance norm',
    url='https://github.com/sz85512678/signature_mahalanobis_knn',
    license='MIT',  # Choose an appropriate license
)