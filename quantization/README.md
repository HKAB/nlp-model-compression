
# Quantization

Experiments for language model compression quantization

## Usage/Examples

```bash
Usage: quantization.py [options]

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

  
