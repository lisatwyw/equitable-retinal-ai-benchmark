# ERAB: Equitable Retinal AI Benchmark

```
src/
├── config.py             # Centralized hyperparameters and task schemas
├── dataset.py            # PyTorch dataset implementation
├── models.py             # Model architectures and development pipeline
├── pipeline.py           # Feature extraction, training, and evaluation
├── main.py               # CLI entry point
└── reproducibility.py    # Fixed random seeds for reproducibility
```


## About

Leveraging the recently proposed [RETFound-Green](https://github.com/justinengelmann/RETFound_Green) representation, which has a 400× lower environmental footprint than its predecessor, we developed models for 12 clinical retinal disease classification tasks using images acquired with the Canon CR-2 camera and evaluated their performance on external images acquired with the Nikon NF5050. This framework enables systematic assessment of camera-associated domain shifts across clinical tasks and modeling configurations.

|     | Train | Validation | Test |  External Test |
| :-- | :-- | :-- | :-- |  :-- |
| $n$ | 7387 | 2191 | 946 | 5639 |
| camera | Canon CR-2 | Canon CR-2 |Canon CR-2   | Nikon NF5050 |
 

<details>
<summary>Details on BRSET and its data dictionary</summary>

```
BRSET contains:

fundus_photos: 16,266 fundus photos images.

labels.csv - database table containing the identifier for each image, demographic information, structural label, diagnosis, and quality parameters labels. Columns are detailed below.

    image_id: image identifier.
    patient_id: patient identifier.
    camera: Retinal camera (Canon CR or NIKON NF5050).
    patient_age: Age of patient in years.
    comorbidities: Free text of self-referred clinical antecedents.
    diabetes_time: Self-referred time of diabetes diagnosis in years.
    insulin_use: Self-referred use of insulin (yes or no).
    patient_sex: Enumerated values: 1 for male and 2 for female.
    exam_eye: Enumerated values: 1 for the right eye and 2 for the left eye.
    diabetes: diabetes diagnosis
    nationality: the patient's nationality.

Anatomical parameters

    optic_disc: Enumerated values: 1 for normal and 2 for abnormal.
    vessels: Enumerated values: 1 for normal and 2 for abnormal.
    macula: Enumerated values: 1 for normal and 2 for abnormal.

Diabetic retinopaty clasification

    DR_ICDR: International Clinic Diabetic Retinopathy classification with enumerated values from 0 to 4.
        0 No retinopathy.
        1 Mild non-proliferative diabetic retinopathy.
        2 Moderate non-proliferative diabetic retinopathy.
        3 Severe non-proliferative diabetic retinopathy.
        4 Proliferative diabetic retinopathy and post-laser status.
    DR_SDRG: Scottish Diabetic Retinopathy Grading Scheme classification with enumerated values from 0 to 4.
        0 No retinopathy.
        1 Mild Background.
        2 Moderate Background.
        3 Severe non-proliferative or pre-proliferative diabetic retinopathy.
        4 Proliferative diabetic retinopathy and post-laser status.

Quality parameters

    focus: enumerated values: 1 for normal and 2 for abnormal.
    illumination: enumerated values: 1 for normal and 2 for abnormal.
    image_field: enumerated values: 1 for normal and 2 for abnormal.
    artifacts: enumerated values: 1 for normal and 2 for abnormal.

Classification parameters

    diabetic_retinopathy- 1 present and 0 absent.
    macular_edema- 1 present and 0 absent.
    scar - 1 present and 0 absent
    nevus - 1 present and 0 absent.
    amd - 1 present and 0 absent.
    vascular_occlusion- 1 present and 0 absent.
    hypertensive_retinopathy - 1 present and 0 absent.
    drusens - 1 present and 0 absent.
    hemorrhage - 1 present and 0 absent.
    retinal_detachment - 1 present and 0 absent.
    myopic_fundus - 1 present and 0 absent.
    increased_cup_disc - 1 present and 0 absent.
    other - 1 present and 0 absent.
```

</details>

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

<br/>
<hr>
<br/>
<a href="https://info.flagcounter.com/ud5H"><img src="https://s01.flagcounter.com/count2/ud5H/bg_FFFFFF/txt_000000/border_CCCCCC/columns_2/maxflags_10/viewers_0/labels_0/pageviews_0/flags_0/percent_0/" alt="Flag Counter" width=20px border="0"></a>

