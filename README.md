[![LICENSE](https://img.shields.io/github/license/jordiclive/Convert-PolyAI-Torch.svg)](https://github.com/jordiclive/Convert-PolyAI-Torch/blob/master/LICENSE)
[![CircleCI](https://circleci.com/gh/jordiclive/Convert-PolyAI-Torch.svg?style=shield)](https://circleci.com/gh/jordiclive/Convert-PolyAI-Torch)
![GitHub issues](https://img.shields.io/github/issues/jordiclive/Convert-PolyAI-Torch.svg)


# First complete Pytorch implmentation of PolyAI's [ConveRT](https://paperswithcode.com/paper/convert-efficient-and-accurate-conversational)
ConveRT: Efficient and Accurate Conversational Representations from Transformers for Pytorch
## Developed By

Jordan Clive(jordan.clive19@imperial.ac.uk). If you have any questions or ideas/improvements please contact me.

## Background

PolyAI built the model in TensorFlow 1—they did not release the code—although, they did release the model object on TensorFlow Hub, so it can be used, fine tuned and the graph/model weights inspected.

This is a Pytorch implementation built from scratch, with inspiration from [codertimo](https://github.com/codertimo/ConveRT-pytorch) who began a similar project but did not get round to implementing the whole model.


## Implementation details

Note: this is only for the single context model for the moment.
...


## Discrepancies (+ possible discrepancies) with original implementation
...


## TODO

- [ ] Finish optimizing on a few batches, efficiency checks (apex fused optimizer etc.)
- [ ] write further training evaluation tests, Continuous Integration tests, artifacts.
- [ ] Write new apache beam Dataflow script, find cheapest way to store on GCP bucket 
- [ ] work out tmp/ file transfer bash scripts during training for logs and checkpoints . GCSFuse
- [ ] more advanced quantization akin to original paper
- [ ] Pretrain on 12 GPU nodes with one Tesla K80 each for 18 hours
- [ ] Do fine tuning downstream benchmarks and compare results

## Training & Logging & Checkpointing

The trainer is in model.py, pass in Pytorch Lightning trainer args if familiar with those, as well as [ConveRTTrainConfig](https://github.com/jordiclive/Convert-PolyAI-Torch/blob/c4ddec5a2ef9c4077d02aeb139029f520d642b9f/src/config.py#L21) arguments. Although a lot of the Lightning had to be overriden, Lightning hooks make this rather simple, so it is well worth putting it in the Lightning framework—so it iseasier to scale up the model, and carry out distributed training and FP16 training. Although the original paper is heavily optimized for floating point 'quantization aware' optimization  eg. 8 bit per embedding parameters with dynamic quantization ranges during training, which I need to look into. (One of the main points of ConveRT is it's quantization). Currently viewing logs in default /lightning_logs with Tensorboard. 


```
python model.py \
    --gpus 8 \
    --precision 16 \
    --batch_size 512 \
    -- distributed_backend 'ddp'
```



## Dataset
PolyAI Reddit data corpus details on how to run on dataflow



## Repository structure

```
├── LICENSE
├── Pipfile
├── README.md
├── data
│   ├── batch_context.pickle		      # example model input object for testing
│   ├── en.wiki.bpe.vs25000.model           # tokenizer model
│   └── sample-dataset.json		     # mini dataset for running overfit batch tests etc.
├── lint.sh
├── requirements-dev.txt
├── requirements.txt
├── setup.cfg
├── src
│   ├── __init__.py
│   ├── config.py                                  #Modelconfig and training config
│   ├── criterion.py                                    
│   ├── dataset.py                                  # prepare dataloaders, with pytorch lightning DataModule
│   ├── lr_decay.py                                 # Lightning callback fn to implement linear warm up of learning rate, followed by cosine annealing
│   ├── model.py                                    # trainer in here, uses Pytorch Lightning for scale                                          
│   └── model_components.py        # All model consituent components, context and reply share Transformer blocks before model forks into distinct projection mlps
└── tests
    ├── __init__.py
    ├── test_dataset.py           
    ├── test_model.py                            # run overfitting on small batch tests etc. check actually trains.
    └── test_model_components.py                 # check shapes etc.
```

## License

Apache License

## Citations

- [ConveRT: Efficient and Accurate Conversational Representations from Transformers](https://arxiv.org/abs/1911.03688)

```bibtext
@misc{1911.03688,
Author = {Matthew Henderson and Iñigo Casanueva and Nikola Mrkšić and Pei-Hao Su and Tsung-Hsien Wen and Ivan Vulić},
Title = {ConveRT: Efficient and Accurate Conversational Representations from Transformers},
Year = {2019},
Eprint = {arXiv:1911.03688},
}
```

## References

The [dataset](https://github.com/jordiclive/Convert-PolyAI-Torch/blob/master/src/dataset.py) preparation code borrows heavily from [codertimo](https://github.com/codertimo). As well as seed code and inspiriation for some of the model components
- [Codertimo's in progress Pytorch conveRT implementation](https://github.com/codertimo/ConveRT-pytorch)
