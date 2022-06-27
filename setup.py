import subprocess
import os
from setuptools import find_packages, setup
from setuptools.command.install import install


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


class CustomInstall(install):

    def run(self):
        install.run(self)
        os.system('pip3 install -r requirements.txt --ignore-installed')
        os.system('pip3 uninstall transformers -y')

        os.system(
            "pip install git+https://github.com/jordiclive/transformers.git@controlprefixes --ignore-installed"
        )
        os.system('pip3 install torchtext==0.8.0 torch==1.7.1')


# in the setup function:

setup(
    name="ControlPrefixes",
    version="0.1.0",
    python_requires='>=3.8',
    description="Code for Control Prefixes for Parameter-Efficient Text Generation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Jordan Clive",
    author_email="jordan.clive19@imperial.ac.uk",
    url="https://github.com/jordiclive/ControlPrefixes",
    download_url="https://github.com/jordiclive/ControlPrefixes.git",
    license="MIT License",
    cmdclass={'install': CustomInstall},
)
