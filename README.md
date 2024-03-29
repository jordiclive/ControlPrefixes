[![LICENSE](https://img.shields.io/github/license/jordiclive/ControlPrefixes.svg)](https://github.com/jordiclive/ControlPrefixes/blob/master/LICENSE)
![GitHub issues](https://img.shields.io/github/issues/jordiclive/ControlPrefixes.svg)

# Control Prefixes for Parameter-efficient Text Generation! 🚅 
This is the implementation of [Control Prefixes for Parameter-efficient Text Generation](https://arxiv.org/abs/2110.08329)

This technique extends Prefix-Tuning, a parameter-efficient technique that tunes prompts at every layers of the transformer and keeps the base LM fixed. Control Prefixes was the first paper to prefix-tune T5 and therefore show how powerful this architecture can be for Structure knowledge graph tasks such as Data-to-Text.

Control Prefixes or LayerControl extends the prefix-tuning framework by having multiple control prefixes for data-point level information. This can inform the model at every layer of attribute-level information and fits into the prefix-tuning framework by sharing the same reparameterizations. Control Prefixes outperforms other methods of conditioning on attribute information that only operate on the token level. 

As a result the technique is state-of-the-art on several datasets:



Data-to-Text:

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/control-prefixes-for-text-generation/data-to-text-generation-on-webnlg-full-1)](https://paperswithcode.com/sota/data-to-text-generation-on-webnlg-full-1?p=control-prefixes-for-text-generation)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/control-prefixes-for-text-generation/data-to-text-generation-on-cleaned-e2e-nlg-1)](https://paperswithcode.com/sota/data-to-text-generation-on-cleaned-e2e-nlg-1?p=control-prefixes-for-text-generation)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/control-prefixes-for-text-generation/text-generation-on-dart)](https://paperswithcode.com/sota/text-generation-on-dart?p=control-prefixes-for-text-generation)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/control-prefixes-for-text-generation/data-to-text-generation-on-webnlg)](https://paperswithcode.com/sota/data-to-text-generation-on-webnlg?p=control-prefixes-for-text-generation)

Abstractive Text Summarization:

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/control-prefixes-for-text-generation/text-summarization-on-x-sum)](https://paperswithcode.com/sota/text-summarization-on-x-sum?p=text-summarization-on-x-sum)

Text Simplification:

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/control-prefixes-for-text-generation/text-simplification-on-asset)](https://paperswithcode.com/sota/text-simplification-on-asset?p=text-simplification-on-asset)



## Developed By

Jordan Clive(jordan.clive19@imperial.ac.uk). If you have any questions or ideas/improvements please contact me.



Installation
------------

```
git clone https://github.com/jordiclive/ControlPrefixes.git
cd ControlPrefixes
pip install .
unzip src/data.zip
unzip src/datatotext/utils.zip
```

Docker 
------
```
docker run --gpus '"all"' --rm -it jordiclive/controlprefixes:main-latest
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

## Citations
------------

- [Control Prefixes for Parameter-efficient Text Generation](https://arxiv.org/abs/2110.08329)

```bibtext
@article{DBLP:journals/corr/abs-2110-08329,
  author    = {Jordan Clive and
               Kris Cao and
               Marek Rei},
  title     = {Control Prefixes for Text Generation},
  journal   = {CoRR},
  volume    = {abs/2110.08329},
  year      = {2021},
  url       = {https://arxiv.org/abs/2110.08329},
  eprinttype = {arXiv},
  eprint    = {2110.08329},
  timestamp = {Fri, 22 Oct 2021 13:33:09 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2110-08329.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

```

- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)

```bibtext
@article{DBLP:journals/corr/abs-2101-00190,
  author    = {Xiang Lisa Li and
               Percy Liang},
  title     = {Prefix-Tuning: Optimizing Continuous Prompts for Generation},
  journal   = {CoRR},
  volume    = {abs/2101.00190},
  year      = {2021},
  url       = {https://arxiv.org/abs/2101.00190},
  archivePrefix = {arXiv},
  eprint    = {2101.00190},
  timestamp = {Thu, 21 Jan 2021 14:42:30 +0100},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2101-00190.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```



