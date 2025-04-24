# XLab

[![PyPI Project](https://img.shields.io/badge/PyPI-xlabml-green?logo=pypi)](https://pypi.org/project/xlabml/)
[![License](https://img.shields.io/badge/License-Apache_2.0-green)](LICENSE)
[![PyTorch Powered](https://img.shields.io/badge/PyTorch-Powered-blue?logo=pytorch)](https://pytorch.org/)
[![Lightning Powered](https://img.shields.io/badge/Lightning-Powered-blue?logo=lightning)](https://lightning.ai/docs/pytorch/stable/)

Transformer Lab - experimental implementations and training of LLMs at scale, using modern techniques.

It features:
- Simple, efficient implementations, and a modular design
- Seamless parallel and reproducible training on multiple GPUs
- Easily configurable architecture modifications and training procedure
- Inference using multiple generation strategies


## Model
A transformer decoder architecture ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762))
with the following modifications:
- Normalization is performed before the transformer sublayers.
  An extra normalization layer is added after the last feedforward sublayer.
  This improved both performance and training stability.
- The GELU activation function is used (performance improvement)
- Dropout is only applied in the attention and feedforward sublayers
  (to the product of the queries and keys, and to the hidden activations, respectively)
  (performance improvement)

Positional encodings are used, because learned positional embeddings degrade performance in the current setup.


## Tokenizer
The following criteria were used to select a tokenization algorithm:
- Able to encode any Unicode string without resorting to "unknown" tokens
- Perfectly reversible (no information loss during preprocessing)

The selected algorithm is BPE (byte pair encoding) from [SentencePiece](https://github.com/google/sentencepiece).
This implementation operates on characters, but is able to encode out-of-vocabulary symbols with bytes.
SentencePiece is mostly reversible, but loses information by replacing spaces with the meta symbol "▁" (U+2581).
To ensure perfect reversibility, during processing this project replaces the meta symbol with a rare sequence,
and unambiguously escapes occurrences of the replacement.
The vocabulary size is 32K tokens, as a reasonable tradeoff between context compression and head size.


## Dataset
[wikimedia/wikipedia, 20231101.en](https://huggingface.co/datasets/wikimedia/wikipedia), split into:
- train: 90%
- val: 5%
- test: 2.5%
- predict: 2.5%

Articles (texts) are chunked into sequences without overlap.
When the (remainder of) a sequence is shorter than the context length, the sequence is padded.
Experiments were conducted with concatenating such sequences,
but this led to reduced training speed and only marginal improvements in validation performance.

## Training
The model was trained on sequences of maximum length 512 tokens.
To speed up training and reduce memory usage,
[FlashAttention-2](https://arxiv.org/abs/2307.08691) and 16-bit mixed precision are used.
To mitigate stability issues, the bfloat16 data type is used, along with gradient clipping by norm 1.0.
The AdamW optimizer is used, with learning rate 3e-4 and weight decay 0.1.
The training ran on a single A100 GPU, with batch size 256, for 3 epochs (144K steps).


## Results
| Version | Checkpoint | Loss (test) | Accuracy (test) |
|---------|------------|-------------|-----------------|
| 0.1     | last       | 3.18        | 40.0%           |
| 0.2     | last       | TBD         | TBD             |


## Getting started
### Installation
XLab can be installed as a Python package, or by cloning this repository.
The former lets you use the models for inference, but to train them you need your own code.
And the latter comes with prebuilt training and inference scripts, as well as configuration.

To install XLab as a Python package, run:
```shell
pip install xlabml
```
Note that this will also install [Lightning](https://lightning.ai/docs/pytorch/stable/) and its dependencies.
Given enough interest, future package versions may come without Lightning, to minimize the number of dependencies.

To install Xlab from this repository, clone it and checkout a release tag (because the main branch may be unstable).
Next, create a virtual environment, and run:
```shell
pip install -r requirements.txt
```

### Obtaining weights
If you need pre-trained weights for inference or fine-tuning,
download the `.pt` file(s) from the project's releases.

### Using in your code
```python
import torch
from xlabml.tokenizer import Tokenizer
from xlabml.datamodules import XLabDataModule
from xlabml.models import XLabModel
from xlabml import inference

# adjust these
checkpoint_path = 'path/to/weights.pt'
prompt = 'april'
device = 'cuda'
limit = 10

tokenizer_path = XLabDataModule.load_from_checkpoint(checkpoint_path, map_location=device).tokenizer_path
tokenizer = Tokenizer.load(tokenizer_path)
model = XLabModel.load_from_checkpoint(checkpoint_path, map_location=device).eval().requires_grad_(False)
inputs = torch.tensor([tokenizer[tokenizer.sos_token]] + tokenizer.encode(prompt), device=model.device)
outputs = inference.sample(
    model, inputs, limit,
    block_size=model.hparams['max_len'],
    eos_class=tokenizer[tokenizer.eos_token]
)
output = tokenizer.decode(outputs.tolist())
```
Note that the resulting *model* object will be a Lightning module.
To decouple the model from Lightning, the underlying PyTorch module is accessible via the *model* attribute.

### Training the model
From the command line, run:
```shell
./xlab.py fit -c conf/xlab.yaml
```
Specifying `-c conf/xlab.yaml` tells the training script to use a larger dataset and model
(the default ones are intended for quick experiments).
This will first download the dataset and tokenizer, and then encode the dataset, which takes about an hour.
If you can't wait, you can try XLab with a smaller dataset and model - simply omit `-c conf/xlab.yaml` everywhere.

The actual training takes about 6 hours per epoch on an A100 GPU.
For hardware with less memory, you may need to modify the configuration and decrease the context size and/or batch size.

By default, the following are saved during training:
- the tokenizer, in the `tokenizers` directory
- the configuration and hyperparameters, in YAML format under `logs/version_*`
- learning curves, in TensorBoard format under `logs/version_*`
- model checkpoints, in `logs/version_*/checkpoints`

You can view the learning curves by running:
```shell
tensorboard --logdir .
```
Keep this running, point your browser to http://localhost:6006/, and click the *Scalars* tab.

### Validation
```shell
./xlab.py validate -c conf/xlab.yaml --ckpt_path PATH
```
where PATH points to a checkpoint, downloaded from this project's releases, or saved during training.
To evaluate on the test set, replace `valudate` with `test` in the above command.

### Inference
For command line inference, run:
```shell
./infer.py CHECKPOINT_PATH "PROMPT"
```
The default generation strategy is multinomial sampling, but greedy search or beam search can also be selected.
Options exist for configuring the algorithms (e.g. temperature, top-k, top-p, beam width, etc).
To see all inference options, run `./infer.py --help`.

### Exporting model weights and hyperparameters
Checkpoints created during training contain not only the model weights,
but also the optimizer state and other information needed to resume training from a saved checkpoint.
This makes the checkpoints 3x larger than the actual model weights.

To export a "clean" checkpoint, containing only the weights and hyperparameters, run:
```shell
./manage.py export-checkpoint CHECKPOINT_PATH
```

### Other actions
#### Validating training data
```shell
./xlab.py validate_data -c conf/xlab.yaml
```
#### Training the tokenizer
First, delete the old tokenizer, and then run:
```shell
./xlab.py train_tokenizer -c conf/xlab.yaml
```
#### Computing dataset statistics
```shell
./xlab.py compute_stats -c conf/xlab.yaml
```

### Configuration
Configuration and hyperparameters are read from multiple sources in the following order:
- defaults in the code
- `conf/defaults.yaml`, implicitly
- YAML files, given with the `-c PATH` option
- command line options

See `conf/xlab.yaml` for the configuration used to train the current release model.
Examples of additional options are in `conf/extra`.
To see the full list, run `./xlab.py --help`.

Hyperparameters are saved in checkpoints and automatically restored when loading the model.


## Generations
**Prompt: April is**  
> April is a municipality in the district of Neuchâtel, within the canton of Lucerne, the canton of Thurgau, Switzerland.
> 
> History
> April racing dinners 
> 
> April generally placed in cultivation with the 'Michael Mayer Cup Dunroipel Limestone Witige Witige Klice Vereu Pino Annarum' on January 2, 1999.
> 
> Geography
> Amongst the top strata of April 2, 1949

**Prompt: The world**  
> The world record in Statistics and Statistics is a record-setting record for any years recorded by the Republic of South Africa. Released September 2014,  racing dataset has had its generally placed in the United States, most of them generally after reports showed that Witwatersrand took the position, and thus concluded a decline in record earnings. It was first held on September 21, 2014, as a measure of the number of admissions of the nation's

**Prompt: Cats and dogs**
> Cats and dogs were described as the "major antagonistic activists". The pro-heroutee activist she was an early Patty Hippolyte Laird and later Frank Hewlett.  She had married the archaeologist Robert Johnson Parr, at the time Dunrood Monument.  Their daughter crowned women winners, Peck Pino and Pedra Acres. It was first reported that of a white man. She first planned to marry a friend, Frank Lloyd Taylor in 1866. She later

All of these can be reproduced with the included inference script, using random seed 42 and limit 100.


## Future work
- add cosine or inverse square root learning rate scheduling after the warmup
- train on progressively longer sequences
- rotary positional embeddings or ALiBi
- RMSNorm
- SwiGLU activation
- train a larger model on a larger and more diverse dataset
- fine-tune for a downstream task
- quantization
- compile the model during training, and for inference
- cache KV pairs in inference, try multi-query/grouped-query attention
