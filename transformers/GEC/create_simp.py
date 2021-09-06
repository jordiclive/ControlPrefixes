from datasets import load_dataset
from pathlib import Path

ds_all = load_dataset("gem", "wiki_auto_asset_turk")

TEST_ONLY = True

splits = ['test_asset','test_turk', 'validation', 'train']
##
for split in splits:

    ds = ds_all[split]

    save_path = Path("data")
    save_path.mkdir(parents=True, exist_ok=True)
    src_path = save_path / f"{split}.source"
    tgt_path = save_path / f"{split}.target"
    ref_path = save_path / f"{split}.references"
    ids_path = save_path / f"{split}.gem_id"
    with open(src_path, 'wt') as src_file, open(tgt_path, 'wt') as tgt_file, open(ref_path, 'wt') as ref_file,open(ids_path, 'wt') as ids_file:
        for i, d in enumerate(ds):
            src, tgt, ref, id = d['source'], d['target'], d['references'], d['gem_id']
            src_len, tgt_len = len(src), len(tgt)

             #remove articles with no summary
            if src_len and tgt_len:
                src = src.replace('\n', '<n>')
                tgt = tgt.replace('\n', '<n>')
                src_file.write(src + '\n')
                tgt_file.write(tgt + '\n')
                #ref_file.write(ref+ '\n')
                ids_file.write(id + '\n')

    print(f"Generated {src_path}")
    print(f"Generated {tgt_path}")
    print(f"Generated {ref_path}")
    print(f"Generated {ids_path}")