from setuptools import setup, find_packages

setup(
    name='hugtokencraft',
    version='0.1.0',
    packages=find_packages(),
    author="Fahim Anjum",
    author_email="dr.fahim.anjum@gmail.com",
    description="HugTokenCraft is a user-friendly Python library that simplifies the process of modifying the vocabulary of a PreTrainedTokenizer from HuggingFace Transformers, making it accessible without additional training.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/MDFahimAnjum/HugTokenCraft",
    include_package_data=True,
    package_data={"": ["example_notebook.ipynb"]},
    python_requires='>=3.9',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=open('requirements.txt').readlines(),
)