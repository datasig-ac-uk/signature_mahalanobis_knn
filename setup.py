from setuptools import setup, find_packages

setup(
    name='signature_mahalanobis_knn',  # Replace with your desired package name
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        # Add other dependencies here
    ],
    dependency_links=['https://github.com/sz85512678/sktime'
    ],
    author='Zhen Shao',
    description='Description of your package',
    url='https://github.com/your_username/your_package_name',
    license='MIT',  # Choose an appropriate license
)