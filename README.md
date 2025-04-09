# XLab
Transformer Lab

## Usage
### Train
```shell
./xlab.py fit -c conf/xlab.yaml
```

### Validate
```shell
./xlab.py validate -c conf/xlab.yaml --ckpt_path PATH
```

## Results
| Version | Checkpoint | Loss (test) | Accuracy (test) |
|---------|------------|-------------|-----------------|
| 0.1     | last       | 3.18        | 40.0            |
