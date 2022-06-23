[![LICENSE](https://)](https:)
![GitHub issues](https://img.shields.io/github/issues/jordiclive/ControlPrefixes.svg)

# Control Prefixes for Parameter-efficient Text Generation! 🚅 
This is the implementation of [Control Prefixes for Parameter-efficient Text Generation](https:)

This technique extends Prefix-Tuning, a parameter-efficient technique that tunes prompts at every layers of the transformer and keeps the base LM fixed. Control Prefixes was the first paper to prefix-tune T5 and therefore show how powerful this architecture can be for Structure knowledge graph tasks such as Data-to-Text.

Control Prefixes or LayerControl extends the prefix-tuning framework by having multiple control prefixes for data-point level information. This can inform the model at every layer of attribute-level information and fits into the prefix-tuning framework by sharing the same reparameterizations. Control Prefixes outperforms other methods of conditioning on attribute information that only operate on the token level. 

As a result the technique is state-of-the-art on several datasets:



Data-to-Text:

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/control-prefixes-for-text-generation/data-to-text-generation-on-webnlg-full-1)](https:)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/control-prefixes-for-text-generation/data-to-text-generation-on-cleaned-e2e-nlg-1)](https:)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/control-prefixes-for-text-generation/text-generation-on-dart)](https:)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/control-prefixes-for-text-generation/data-to-text-generation-on-webnlg)](https://google.com)

Abstractive Text Summarization:

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/control-prefixes-for-text-generation/text-summarization-on-x-sum)](https:)

Text Simplification:

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/control-prefixes-for-text-generation/text-simplification-on-asset)](https:)



## Developed By

Jordan Clive(jordan.clive19@imperial.ac.uk). If you have any questions or ideas/improvements please contact me.



Installation
------------

```
cd ControlPrefixes
bash setup.sh
unzip src/data.zip
unzip src/datatotext/utils.zip
```
Usage
-----

Data-to-Text datasets with conditional data-point attribute information is provided at [`src/data/`](src/data/processed/). For XSum, sample files are provided. 

Each model config is contained in the config folder. E.g. For data-to-text at [`src/datatotext/configs/`](src/datatotext/configs/). Edit the data and output directory paths, gpus. To use distributed training refer to the Pytorch-Lightning docs. 

To run training, e.g. for the original 2017 WebNLG.
```
$ cd src/datatotext 
$ python read_yaml.py configs/webnlg17_config.yaml
```

## License

Apache License




