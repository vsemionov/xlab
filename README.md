# XLab

[![PyPI Project](https://img.shields.io/badge/pypi-xlabml-blue?logo=pypi)](https://pypi.org/project/xlabml/)
[![PyTorch Powered](https://img.shields.io/badge/pytorch-powered-blue?logo=pytorch)](https://pytorch.org/)
[![Lightning Powered](https://img.shields.io/badge/lightning-powered-blue?logo=lightning)](https://lightning.ai/docs/pytorch/stable/)

Transformer Lab - experimental implementations and training of transformer models at scale, using modern techniques.

It features:
 - Simple, efficient implementations
 - Parallel and reproducible training on multiple GPUs and nodes
 - Configurable architecture modifications and training procedure
 - Inference using various generation strategies


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

To install Xlab from this repository, clone it, create a virtual environment, and run:
```shell
pip install -r requirements.txt
```

### Obtaining weights
If you need pre-trained weights for inference or fine-tuning,
download the `.pt` file(s) from the project's releases.

### Using in your code
```python
import torch
from xlabml.datamodules import XLabDataModule
from xlabml.models import XLabModel
from xlabml import inference

# adjust these
checkpoint_path = 'path/to/weights.pt'
prompt = 'april'
device = 'cuda'
limit = 10

tokenizer = XLabDataModule.load_from_checkpoint(checkpoint_path, map_location=device).tokenizer
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
This will also download and pre-process the dataset, as well as train the tokenizer, which takes about 2 hours total.
The actual training takes about 10 hours per epoch on an A100 GPU.
For hardware with less memory, you may need to modify the configuration and decrease the context size and/or batch size.
By default, the learning curves are saved in TensorBoard format, and you can monitor them by running:
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
The default generation strategy is multinomial sampling, but beam search can also be selected.
Options exist for configuring the algorithms (e.g. temperature, top-k, top-p, beam width, etc).
To see all inference options, run `./infer.py --help`.

### Exporting model weights and tokenizer vocabulary
Checkpoints created during training contain not only the model weights,
but also the optimizer state and other information needed to resume training from a saved checkpoint.
This makes the checkpoints 3x larger than the actual model weights.
To export a "clean" checkpoint, containing only the weights and vocabulary, run:
```shell
./manage.py export-checkpoint CHECKPOINT_PATH
```

### Configuration
All configuration and hyperparameters are exposed in YAML files, passed to the training/validation script.
Hyperparameters are saved in checkpoints and automatically restored when loading.
The default settings are in `conf/defaults.yaml`.
Additional YAML configuration can be specified with the `-c PATH` option.
See `conf/xlab.yaml` for the configuration used to train the current release model.
Additional options (or overrides of the above configuration) can be specified on the command line.
To see the full list, run `./xlab.py --help`.


## Model
A decoder-only transformer architecture ([Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)) with the following modifications:
 - Normalization is performed before the transformer sublayers.
   An extra normalization layer is added after the last feedforward sublayer.
   This improved both performance and training stability.
 - The GELU activation function is used (performance improvement)
 - Dropout is only applied in the attention and feedforward sublayers
   (to the product of the queries and keys, and to the hidden activations, respectively)
   (performance improvement)

Positional encodings are used, because learned positional embeddings degrade performance in the current setup.


## Tokenizer
The implementation from [torchtext](https://github.com/pytorch/text),
which lowercases the input text, strips punctuation, and yields tokens between whitespace boundaries.
The vocabulary is built from the 32K tokens with the highest frequencies across the training set.
Tokens matching special values (e.g. *&lt;unk&gt;* and *&lt;pad&gt;*) are escaped
in order to avoid misinterpretation and improve reversibility.


## Dataset
[wikimedia/wikipedia, 20231101.en](https://huggingface.co/datasets/wikimedia/wikipedia), split into:
 - train: 90%
 - val: 5%
 - test: 2.5%
 - predict: 2.5%

Articles (texts) are chunked into sequences with 50% overlap.


## Training
The model was trained on sequences of maximum length 256.
To speed up training and reduce memory usage, 16-bit mixed precision is used.
To mitigate stability issues, the bfloat16 data type is used, along with gradient clipping by norm 1.0.
The AdamW optimizer is used, with learning rate 3e-4 and weight decay 0.1.
The training ran on a single A100 GPU, with batch size 256, and was stopped after 4 epochs (440K steps after 2 days).


## Results
| Version | Checkpoint | Loss (test) | Accuracy (test) |
|---------|------------|-------------|-----------------|
| 0.1     | last       | 3.18        | 40.0%           |


## Generations
**Prompt: april is**  
*april is the second album of the band dead from dead . it was released on february 15 , 2007 to predominantly negative reviews . it features several varied development and music styles . the album peaked at number only in the &lt;unk&gt; area , thus breaking the album ' s passage into a new york times best seller . it was also ranked as the fourth best album of 2007 by music critic roger ebert . track listing spirit of the line ( ' ticket to ride ' ) meant to awake into us quarter-finals phenomenon memory of misery activates fire*

**Prompt: the world**  
*the world bowling championships were a women ' s national bowling championships organized in walnut street , new york city to open in 1980 . it is the world development and development archive event at the world bowling hall of fame , located in oak park , wisconsin . initially developed as a bowling and tennis track for workers , it was expanded into winter surface courses as a grassroots project to preserve open library needed oral materials to support the expanded bowling programs for disabled adults . medal summary results by round matches us quarter-finals u . s . championship top*

**Prompt: cats and dogs**  
*cats and dogs ( , , ) is a 1986 indian malayalam-language drama film directed by m . k . raman nair and produced by the film production company &lt;unk&gt; development . it stars &lt;unk&gt; &lt;unk&gt; and &lt;unk&gt; , while &lt;unk&gt; in the lead roles , &lt;unk&gt; &lt;unk&gt; , and muhammed in three members of the comedy team . the film is a remake of the 1989 hindi film &lt;unk&gt; . it was remade in telugu as oral cough . inscription the film was released on 6 april 1986 in kerala . plot a criminal named &lt;unk&gt; visits rani ( &lt;unk&gt; &lt;unk&gt; )*

All of these can be reproduced with the included inference script, using random seed 42 and limit 100.


## Future work
 - use a BPE tokenizer, e.g. from tiktoken or sentencepiece
 - group sequences of similar lengths in the same training batches
 - add learning rate scheduling (e.g. cosine with warmup, or reduce lr on plateau)
 - rotary positional embeddings
 - RMSNorm
 - SwiGLU activation
 - increase the maximum context length, e.g. via gradient checkpointing
 - train a larger model on a larger and more diverse dataset
 - fine-tune for a downstream task
 - quantization
 - compile the model during training, and for inference
 - cache KV pairs in inference, try multi-query/grouped-query attention
