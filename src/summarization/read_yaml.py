import yaml
import os
import fire


def run_experiment(yaml_file):
    with open(yaml_file, "r") as stream:
        parsed_yaml = yaml.safe_load(stream)

    args = ""
    for arg, value in parsed_yaml.items():
        args += f"--{arg} {value} "

    os.system(f"python finetune.py {args}")


if __name__ == "__main__":
    fire.Fire(run_experiment)
