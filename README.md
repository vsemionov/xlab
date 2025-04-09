# XLab
Transformer Lab

XLab is a personal experiment to learn implementing and training transformer models using modern techniques.
It supports:
 - Configurable model sizes, architecture modifications, datasets, tokenizer settings, and training procedure
 - Parallel training on multiple GPUs and nodes
 - Inference using different algorithms / sampling strategies (e.g. temperature, top-p, top-k, beam search)


## Getting started
### Environment setup
First, create a virtual environment. Install dependencies by running:
```shell
pip install -r requirements.txt
```

### Download weights
If you want to run inference using pre-trained weights, download the `.pt` file(s) from this project's releases.

### Train the model
You can skip training if you only want to run inference using pre-trained weights.
To train the model, run the following command:
```shell
./xlab.py fit -c conf/xlab.yaml
```
Specifying `-c conf/xlab.yaml` tells the training script to use a larger dataset and model
(the default ones are intended for quick experiments).
This will also download and pre-process the dataset, as well as train the tokenizer, which takes about 2 hours.
The actual training takes about 10 hours per epoch on an A100 GPU.
For models with less memory, you may need to modify the configuration and decrease the context size and/or batch size.
By default, the learning curves are saved in TensorBoard format, and you can monitor them by running:
```shell
tensorboard --logdir .
```
Keep this running, point your browser to http://localhost:6006/, and click the *Scalars* tab.

### Validate (optional)
```shell
./xlab.py validate -c conf/xlab.yaml --ckpt_path PATH
```
where PATH points to a checkpoint, downloaded from this project's releases, or saved during training.
To evaluate on the test set, replace `valudate` with `test` in the above command.

### Inference
For basic inference using multinomial sampling, run:
```shell
./infer.py [OPTIONS] CHECKPOINT_PATH "PROMPT"
```
To see other inference options, run `./infer.py --help`.

### Exporting model weights and tokenizer vocabulary
Checkpoints created during training contain not only the model weights,
but also the optimizer state and other information needed to resume training from a saved checkpoint.
This makes the checkpoints 3x larger than the actual model weights.
To export a "clean" checkpoint, containing only the weights and vocabulary, run:
```shell
./manage.py export-checkpoint CHECKPOINT_PATH
```

### Use in your code
```python
import torch
from xlab.datamodules import XLabDataModule
from xlab.models import XLabModel
from xlab import inference

# adjust these
checkpoint_path = 'logs/version_0/checkpoints/last.ckpt'
prompt = 'april'
device = 'cuda'
limit = 10

tokenizer = XLabDataModule.load_from_checkpoint(checkpoint_path, map_location=device).tokenizer
model = XLabModel.load_from_checkpoint(checkpoint_path, map_location=device).eval().requires_grad_(False)
inputs = torch.tensor([tokenizer[tokenizer.sos_token]] + tokenizer.encode(prompt), device=model.device)
outputs = inference.sample(
    model, inputs,
    output_length=limit, block_size=model.hparams['max_len'],
    eos_class=tokenizer[tokenizer.eos_token]
)
output = tokenizer.decode(outputs)
```

### Configuration
All configuration and hyperparameters are exposed in YAML files, passed to the training/validation script.
Hyperparameters are saved in checkpoints and automatically restored when loading.
The default settings are in `conf/defaults.yaml`.
Additional YAML configuration can be specified with the `-c PATH` option.
See `conf/xlab.yaml` for the configuration used to train the current release model.
Additional options (or overrides of the above configuration) can be specified on the command line.
To see the full list, run `./xlab.py --help`.

## Dataset
[wikimedia/wikipedia, 20231101.en](https://huggingface.co/datasets/wikimedia/wikipedia), split into:
 - train: 90%
 - val: 5%
 - test: 2.5%
 - predict: 2.5%


## Results
| Version | Checkpoint | Loss (test) | Accuracy (test) |
|---------|------------|-------------|-----------------|
| 0.1     | last       | 3.18        | 40.0            |
