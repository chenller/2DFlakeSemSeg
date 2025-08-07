from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mmsegalext',  # The name of the package
    version='0.1',  # The version of the package
    description='An active learning extension library for mmseg',  # A short summary of the package
    long_description=long_description,  # Long description read from the the readme file
    long_description_content_type='text/markdown',  # The content type of the long description
    author='chanller',  # Author's name
    author_email='your.email@example.com',  # Replace with the actual author's email address
    url='https://github.com/chanller/mmseg-activelearning-extension',  # Replace with the actual URL of the project
    packages=find_packages(),  # Automatically find all packages in the specified directory
    python_requires='>=3.0',  # Minimum required Python version
)
