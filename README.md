# pytorch-lightning-sandbox

## Requrement
- rye
  - document: https://rye-up.com/guide/installation/
  - installation:
```bash
curl -sSf https://rye-up.com/get | bash
```

## Setup
```bash
rye sync
```

## Train
```bash
. .venv/bin/activate
python -m run.train
```

## Inference
```bash
. .venv/bin/activate
python -m run.inference
```
