[![LICENSE](https://img.shields.io/github/license/jordiclive/Convert-PolyAI-Torch.svg)](https://github.com/jordiclive/Convert-PolyAI-Torch/blob/master/LICENSE)
[![CircleCI](https://circleci.com/gh/jordiclive/Convert-PolyAI-Torch.svg?style=shield)](https://circleci.com/gh/jordiclive/Convert-PolyAI-Torch)
![GitHub issues](https://img.shields.io/github/issues/jordiclive/Convert-PolyAI-Torch.svg)


# Control Prefixes 
This is the implementation of Control Prefixes.

A method based on Prefix-Tuning, which incorporates conditional input-dependent prompts. This method is at the intersection of prompt learning
and controlled generation.
## Developed By

Jordan Clive(jordan.clive19@imperial.ac.uk). If you have any questions or ideas/improvements please contact me.


## Training & Logging & Checkpointing

```
python transformers/webnlg/finetune_2.py 
    --warmup_steps 2000 \
    --num_train_epochs 30 \
    --num_sanity_val_steps 4 \
    --m_prefix_len 2 \
    --preseqlen 48 \
    --train_batch_size 6 \
    --eval_batch_size 3 \
    --gradient_accumulation_steps 16 \
    --check_val_every_n_epoch 1 \
    --learning_rate 1.5e-05     
```




## License

Apache License

## Citations

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


