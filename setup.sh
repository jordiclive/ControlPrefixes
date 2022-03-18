
pip3 install -r requirements.txt --ignore-installed
cd src/transformers
pip3 uninstall transformers -y
pip3 install -e . --ignore-installed
pip3 install torchtext==0.8.0 torch==1.7.1