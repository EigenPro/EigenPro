from setuptools import setup, find_packages

import eigenpro


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
    
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    
setup(
    name='eigenpro',
    version=eigenpro.__version__,
    author='Amirhesam Abedsoltan, Siyuan Ma, Parthe Pandit',
    author_email='aabedsoltan@ucsd.edu, siyuan.ma9@gmail.com, ' +
                 'parthe1292@gmail.com',
    description='Fast solver for Kernel Regression using GPUs with linear ' +
                'space and time complexity',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/EigenPro/EigenPro',
    project_urls = {
        "Bug Tracker": "https://github.com/EigenPro/EigenPro/issues"
    },
    license='Apache-2.0 license',
    packages=find_packages(),
    install_requires=requirements,
)
