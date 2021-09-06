from datasets import load_dataset
from pathlib import Path
dataset =

splits = ['test', 'validation', 'train']
for split in splits:

    ds = dataset[split]

    save_path = Path("/content/gdrive/MyDrive/xsum_datasets")
    save_path.mkdir(parents = True, exist_ok = True)
    src_path = save_path / f"{split}.source"
    tgt_path = save_path / f"{split}.target"
    with open(src_path, 'wt') as src_file, open(tgt_path, 'wt') as tgt_file:
        for i, d in enumerate(ds):
            src, tgt = d['document'], d['summary']
            src_len, tgt_len = len(src), len(tgt)

            if src_len and tgt_len:
                src = src.replace('\n', '<n>')
                tgt = tgt.replace('\n', '<n>')
                src_file.write(src + '\n')
                tgt_file.write(tgt + '\n')

