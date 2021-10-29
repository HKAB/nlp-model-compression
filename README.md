
# NLP Compression

Experiments for language model compression


## Features

- Quantization
- Movement pruning for RoBERTa (source code is [huggingface](https://github.com/huggingface/transformers/tree/master/examples/research_projects/movement-pruning))

  
## Results

Pretrained phoBERT on NER task: [Google drive](https://drive.google.com/file/d/1-6unZYakOL6z6rCe3N31uLmEM5roQ5Vp/view?usp=sharing)

Pretrained phoBERT on POS task: [Google drive](https://drive.google.com/file/d/1RTEPnz9gtrNxjfaaIxQDnozxFA_5pi8Z/view?usp=sharing)

Report: [Google drive PDF](https://drive.google.com/file/d/1rdP5A6FgLSIgf_1d3703pBtdPzIPRd0M/view?usp=sharing)
## Usage/Examples

```bash
Usage: main.py [options]

Options:
  -h, --help            show this help message and exit
  -b BATCH, --batch=BATCH
                        Batch size. Default: 32
  -t TASK, --task=TASK  Type of task: 'n' (for NER) or 'p' (for POS). Default: p
  -e EPOCHS, --epochs=EPOCHS
                        Number of epochs. Default: 30
  -m MODE, --mode=MODE  train/test/measure_infer mode. Default: train

    For training:               python main.py -m train -t [TASK] -b [BATCH] -e [EPOCH]
    For test:                   python main.py -m test -t [TASK] -b [BATCH] -e [EPOCH]
    For measure inference time: python main.py -m measure_infer -t [TASK]
        (Make sure you change the MEASURE_MODEL_PATH constant to the right model on results folder)

```

  
