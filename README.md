# stanford-babylm

This is the shared codebase for the Stanford BabyLM team. It is basically a dressed-up version of Huggingface's standard [`run_clm.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py) script, which does causal (decoder-only) language modeling on a set of training files and evaluates perplexity on a set of validation files.

It is "dressed up" in the sense that

1. we have switched to Hydra for experiment (see below),
2. we have refactored some of the code into separate files for cleanliness,
3. we have slightly improved Weights and Biases logging.

# Setup

Please follow these innstructions for setup so we can have a consistent development environment and package versions.

## 1. Create conda env

I recommend creating a new conda environment. For consistency let's all use `python3.8`.

```
conda create --name babylm python=3.8
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

Whenever you make a commit, if your code is not formatted properly, pre-commit
should automatically format your code. Then you need to add the new files and
repeat the commit message.

If you use VSCode for development I also highly recommend using the
[black](https://marketplace.visualstudio.com/items?itemName=ms-python.black-formatter)
and [isort](https://marketplace.visualstudio.com/items?itemName=ms-python.isort)
plugins, auto-formatting upon save. You can read a tutorial here: https://cereblanco.medium.com/setup-black-and-isort-in-vscode-514804590bf9

# Usage

Example command:

```
python -m src.run_clm +model=gpt2-small +dataset=babylm_10M
```

This runs GPT2-Small architecture on all of the 10M training files, with eval on all of the dev files. It will log a run to
`wandb.ai/<YOUR_USERNAME>/stanford-babylm/groups/debug-gpt2-small-babylm_10M`.

If you use VSCode, there is a launch config in `.vscode/launch.json` which runs
this in the VSCode debugger.

## About arguments and experiment configuration

This codebase uses [Hydra](https://github.com/facebookresearch/hydra) to manage
and configure experiments. Hydra is a little more complicated than `argparse`,
which you may be familiar with, but offers much more flexibility for running and
managing complex configurations.

The key idea of hydra is that, using `argparse`, we rarely ever need to change a
single `argparse` flag. For example imagine if your command was

```
python -m src.run_clm --model=gpt2
```

And you wanted to use GPT2-Medium instead. But GPT-2 medium is bigger, so you
might need to change the batch size to fit it on your GPU. Maybe you want to add
gradient accumulation steps; maybe you want to save less often since the amount
of data per step has changed. So now your command looks like this:

```
python -m src.run_clm --model=gpt2-medium \
    --per_device_train_batch_size=8 \
    --per_device_eval_batch_size=8 \
    --eval_steps=200 \
    --gradient_accumulation_steps=2
```

This gets super annoying to type out and hard to remember. The idea of `hydra`
is not only have the option of changing individual arguments via the CLI like
the above; you can also save **groups** of changes into a YAML configuration
file, and apply **groups** of changes at once. So for example you might make a
file, `src/conf/model/gpt2-medium.yaml`, which looks like:

```yaml
# @package _global_
model: gpt2-medium
per_device_train_batch_size: 8
per_device_eval_batch_size: 8
eval_steps: 200
gradient_accumulation_steps: 2
```

And now you can run your GPT2-Medium experiment by just writing

```
python -m src.run_clm +model=gpt2-medium
```

You'll see that the example run above uses two config groups: a `dataset` group
and a `model` group. You can access the corresponding YAML files in
`src/conf/dataset` and `src/conf/model`, respectively. There is a default config
as well: `src/conf/config.yaml`. The values in the default config get read
first; then the additional configs (applied a la `+model=`) override the default
values. If you change the default config, make sure it is a change that you
think will should sensibly be set for all further experiment runs (e.g. a
default setting in case we add some new architecture). Otherwise, add new
changes in a separate config group.

Another benefit is that unlike argparse, Hydra supports rich hierarchical/nested config structure. So you can have a set of training arguments in your YAML file like

```yaml
model:
  model_name_or_path: gpt2-small

training:
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 4
```
and these are kept in separate namespaces, and you can use `args.model.model_name_or_path` or `args.training.per_device_train_batch_Size` to refer to the relevant args in your script.

Feel free to add more YAML files or even create new config groups. This will
come in handy as we do a lot of experimentation!

Lastly, Hydra also supports running *sweeps* over arguments automatically, [via command line](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run/). I'll talk about this later if we get to it.

## Weights and Biases

We use Weights and Biases for (super convenient!) experiment logging and management. Follow the setup instructions above to get WandB setup.

Weights and Biases config is slightly different from Huggingface default. Take a
look at the `WandBArguments` class in `arguments.py`, and correspondingly, the
`wandb:` set of arguments in `src/conf/config.yaml`, which look like:

```yaml
wandb:
  # Set this to false here or via command line (e.g. `wandb.log=false`) to disable wandb logging.
  log: true
  # The name of the wandb project under your account.
  project: stanford-babylm
  # The name of the group that a run will fall under. This intimidating syntax
  # just interpolates values from other config settings set during your run. For
  # example ${hydra:runtime.choices.model} is the +model= command you specified;
  # likewise via dataset. `wandb.tag` is a key that is not set here but by
  # default is set to "debug". You might tag experiments with the current date,
  # or a name, to remind yourself of which experiments are which. This lets you
  # change the name of the group while still having the auto-generated group
  # name parts (model/dataset).
  group: ${wandb.tag}-${hydra:runtime.choices.model}-${hydra:runtime.choices.dataset}
  # Groups carry groups of runs; this is the name of the individual run, which is the group + a seed.
  name: ${wandb.group}-run-${training.seed}
```

The comments are self explanatory, but basically, if you want to tag an
experiment so that it shows up as a separate group, you might do something like

```
python -m src.run_clm +model=gpt2-small +dataset=babylm_10M wandb.tag=test-new-architecture
```

which generates a run at `wandb.ai/<YOUR_USERNAME>/stanford-babylm/groups/test-new-architecture-gpt2-small-babylm_10M`.

Note the lack of `+` when setting individual keys, versus the `+` needed when
applying an entire YAML file to override configs.

## Adding arguments

To add new arguments, you will want to (1) add the corresponding argument, with its
type, to the corresponding dataclass object (e.g. `DataTrainingArguments` or
`TrainingArguments`) in `arguments.py`; then (2) you can specify it via command
line or in a config file. This uses Hydra's [Structured
Configs](https://hydra.cc/docs/tutorials/structured_config/intro/) settings to
enable static typechecking.
