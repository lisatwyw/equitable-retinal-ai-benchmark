# retinal-camera-domain-shift

```
src/
├── config.py         # Centralized hyperparameters & task schemas
├── dataset.py        # PyTorch Dataset implementation
├── models.py         # Vision backbone & Classifier factory
├── pipeline.py       # Feature extraction, training, evaluation
└── main.py           # CLI entry point for execution
```


## About

Leveraging the recently proposed [RETFound-Green](https://github.com/justinengelmann/RETFound_Green) representation, which has a 400× lower environmental footprint than its predecessor, we developed models for 12 clinical retinal disease classification tasks using images acquired with the Canon CR-2 camera and evaluated their performance on external images acquired with the Nikon NF5050. This framework enables systematic assessment of camera-associated domain shifts across clinical tasks and modeling configurations.

|     | Train | Validation | Test |  External Test |
| :-- | :-- | :-- | :-- |  :-- |
| $n$ | | | | | |
