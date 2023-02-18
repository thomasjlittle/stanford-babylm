# stanford-babylm

# Setup

Please follow these innstructions for setup so we can have a consistent development environment and package versions.

## 1. Create conda env

I recommend creating a new conda environment. For consistency let's all use `python3.8`.

```
conda create --name babylm python-3.8
conda activate babylm
```

Now, first install a version of PyTorch that is compatible with your system from
[pytorch.org](https://pytorch.org/). Try to use at least `1.12.0`.

Then install extra requirements from `requirements.txt`. Torch is not included
since installation differs depending on the machine.

```
pip install -r requirements.txt
```

## 2. Create cache dir

**Next**, make a hidden `.cache` dir in the root directory.

```
mkdir .cache
```

Make sure this is somewhere that has a lot of space. If needed, symlink the directory.

## 3. Download babylm data

Run `./download_data.sh` in home dir to download BabyLM data to
`./babylm_data`.

## 4. Create a Weights and Biases account

Follow instructions here: https://docs.wandb.ai/quickstart

## 5. Set up VSCode/pre-commit

We're going to use `black` and `isort`, python formatters, to automatically
format code whenever we push to maintain consistent style. `pre-commit` should
have been installed from `requirements.txt`. Now do:

```
pre-commit install
```

Whenever you make a commit, pre-commit should automatically format your code.

If you use VSCode for development I also highly recommend using the
[black](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
and [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort)
plugins, auto-formatting upon save.

# Usage

Example command:

```
python -m src.run_clm +model=gpt2-small +dataset=babylm_10M
```

This runs GPT2-Small architecture on all of the 10M training files, with eval on all of the dev files.

If you use VSCode, there is a launch config in `.vscode/launch.json` which runs
this in the VSCode debugger.

## About arguments and experiment configuration

This codebase