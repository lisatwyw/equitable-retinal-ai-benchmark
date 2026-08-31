# ERAB: Equitable Retinal AI Benchmark

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
| $n$ | 7387 | 2191 | 946 | 5639 |
 


<details>

<summary>Reproducing the experimental results</summary> 

## Installation

```bash
git clone https://github.com/lisatwyw/equitable-retinal-ai-benchmark.git
cd equitable-retinal-ai-benchmark
pip install -r requirements.txt
```


</details>


<details>

<summary>Details on the experimental setup</summary>

 
### Input Resolutions

To evaluate the effect of input resolution on cross-camera performance, images were evaluated at four input resolutions:

* **224 × 224**
* **392 × 392** — the input resolution used during RETFound-Green pretraining
* **672 × 672**
* **952 × 952**

For each resolution, the same resizing and normalization procedure was applied across Canon CR-2 and Nikon NF5050 images. This allowed us to assess whether increasing input resolution mitigates camera-associated performance differences.

</details>
