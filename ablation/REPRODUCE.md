# Reproducibility Guide

This document describes how to build, test, package, and reproduce the experiments in this repository across HPC systems such as Rorqual and Narval.

The project uses:

- Git for source-code version control
- A Python virtual environment for development
- Apptainer/Singularity for reproducible execution
- SLURM for HPC job submission
- GPU acceleration through the host cluster
- Configuration files to define individual experiments

The goal is that a researcher can clone this repository on another compatible HPC cluster and reproduce the experiments without manually recreating the original Python environment.

<details>
    <summary>1. Repository structure</summary>


The recommended repository structure is:
```
project/
├── README.md
├── REPRODUCE.md
├── requirements.txt
├── Dockerfile
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── metrics.py
│
├── configs/
│   ├── variant1.yaml
│   └── variant2.yaml
│
├── scripts/
│   ├── train.sh
│   └── evaluate.sh
│
├── slurm/
│   ├── rorqual.slurm
│   └── narval.slurm
│
├── containers/
│   └── retinal-ablation.sif
│
├── checkpoints/
├── results/
└── data/
```

Large datasets and experiment outputs should generally not be committed to Git.

</details>

<details>
 
<summary> 2. Development environment on the original HPC system</summary>

The first step is to create an isolated Python environment.

On the development cluster, inspect the available software modules:

```module avail```


Identify the Python and CUDA/PyTorch environment appropriate for the cluster.

Load the required modules according to the cluster's current documentation.

For example:

```module load python```


Do not copy module names blindly between clusters. Rorqual and Narval may provide different module versions.

## 3. Create the virtual environment

Create a project-specific virtual environment:

```python -m venv .venv```


Activate it:

```source .venv/bin/activate```


Verify:

```
which python
python --version
```

Upgrade packaging tools:

```pip install --upgrade pip setuptools wheel```

</details>
    
<details>
    
<summary>4. Install the project dependencies    </summary>

Install the dependencies required by the project.

During development, install packages normally:
```
pip install torch
pip install timm
pip install numpy
pip install pandas
pip install scikit-learn
pip install pillow
```

Add any other dependencies required by the project.

Once the environment is working, record the exact installed versions:
```
pip freeze > requirements.txt
```

Review requirements.txt before committing it.

The purpose of this file is to document the software environment used to develop the project.

5. Verify the GPU environment

Before running a full experiment, verify that PyTorch can see the GPU:
```
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
PY
```

Do not proceed with a long experiment until GPU availability has been verified.

</details>


<details>
6. Test the project outside the container

Before creating a container, make sure the ordinary Python environment works.

For example:
```      
python src/train.py --config configs/variant1.yaml
```

Run a small test experiment first.

Confirm that:

- the dataset can be found;
- images can be loaded;
- the model initializes;
- the GPU is detected;
- one training batch completes;
- validation completes;
- checkpoints are written;
- metrics are generated.

Only after this works should the environment be containerized.

7. Important: separate software from cluster-specific paths

Do not hard-code Rorqual-specific paths inside Python source files.

Avoid:
```    
DATA_DIR = "/project/def-someuser/odir"
```

Instead, make paths configurable:
```
python src/train.py \
    --config configs/variant1.yaml \
    --data-dir "$DATA_DIR"
```

or define the data location in a cluster-specific configuration.

The Python code should not need to change when moving from Rorqual to Narval.

8. Create the container definition

The ```Dockerfile``` is the canonical description of the software environment.

A simplified example is:
```
FROM python:3.11-slim

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY src /workspace/src
COPY configs /workspace/configs

ENV PYTHONUNBUFFERED=1

CMD ["python", "src/train.py"]
```

The exact base image and PyTorch installation should be selected according to the project's CUDA requirements.

The Dockerfile should contain the software environment, not the dataset.

9. Build the container

Containers can be built using a system capable of building Docker/OCI images.

For example:
```
docker build -t retinal-ablation:1.0 .
```

Check that the image was created:
```
docker images
```

The version number should correspond to a meaningful project release.

For example:
```
retinal-ablation:1.0
retinal-ablation:1.1
retinal-ablation:2.0
```

Do not silently replace the container used for a published experiment.

</details>


<details>

    10. Convert/publish the container for HPC use

HPC systems commonly use Apptainer rather than Docker for execution.

The ```Docker/OCI``` image can be converted to an Apptainer image:
```
retinal-ablation.sif
```

The exact build command depends on where the image is being built and the cluster's policies.

The important result is a versioned image:
```
retinal-ablation-1.0.sif
```

Keep the image associated with the corresponding Git commit/release.

For example:
```
Git tag:       v1.0
Container:     retinal-ablation-1.0.sif
```

This makes it possible to identify exactly which software environment produced a result.

11. Test the Apptainer container

Before distributing the container, test it on the original HPC system.

For example:
```
apptainer exec --nv \
    retinal-ablation-1.0.sif \
    python -c "import torch; print(torch.__version__)"
```

Then test GPU access:
```
apptainer exec --nv \
    retinal-ablation-1.0.sif \
    python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

The ```--nv``` option exposes the NVIDIA GPU/driver environment from the HPC host to the container.

The container does not contain the physical GPU or host kernel driver.

12. Test the actual experiment inside the container

Run a small experiment:
```
apptainer exec --nv \
    retinal-ablation-1.0.sif \
    python src/train.py \
    --config configs/variant1.yaml
```

Compare the result with the experiment performed using the virtual environment.

The results should be consistent within the expected randomness of the training procedure.

13. Store large files outside Git

Do not commit:
```
*.pt
*.pth
*.ckpt
*.sif
large datasets
large experiment outputs
```

unless there is a specific reason to do so.

Large files should generally be stored in project/scratch storage or a suitable research data repository.

The Git repository should contain the information required to obtain them.

14. Reproducing the project on another HPC cluster

A new researcher does NOT need to recreate the original Rorqual virtual environment if the Apptainer image is provided.

They should start by cloning the repository:
```
git clone <REPOSITORY_URL>
cd <PROJECT_NAME>
```

Check out the appropriate release:
```
git checkout v1.0
```

This is important because the source code should match the container version.

15. Obtain the exact container

Obtain:
```
retinal-ablation-1.0.sif
```

from the project's documented container location.

Place it somewhere accessible on the cluster, for example:
```
$PROJECT/containers/retinal-ablation-1.0.sif
```

The container should match the Git release being reproduced.

For example:
```
Git v1.0
    +
retinal-ablation-1.0.sif
```

Do not use a newer container unless you are intentionally reproducing a newer experiment.

16. Check Apptainer on the new cluster

On the new HPC cluster:
```
module avail
```

Find the available Apptainer module according to the cluster's documentation.

Load it:
```
module load apptainer
```

Verify:
```
apptainer --version
```

The researcher does not need Docker simply to execute the Apptainer image.

17. Check GPU access

Request a GPU through the cluster's scheduler.

The exact SLURM options differ between clusters.

Once inside a GPU allocation, test:
```
apptainer exec --nv \
    retinal-ablation-1.0.sif \
    python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

If CUDA is unavailable, do not start the full experiment.

The problem may be related to:

- GPU allocation;
- cluster modules;
- Apptainer GPU integration;
- host NVIDIA driver compatibility;
- container CUDA/PyTorch compatibility.

18. Configure the dataset path

The dataset does not need to be inside the container.

For example:
```
export DATA_DIR=/project/<account>/datasets/ODIR
```

Verify that the expected files exist:
```
ls "$DATA_DIR"
```

The dataset directory structure should follow the structure documented by this project.

Researchers should not modify the Python source code simply because the dataset is stored at a different location.

19. Running the experiment on Narval/Rorqual/etc.

The experiment should be launched using the cluster-specific SLURM script.

For example:
```
sbatch slurm/narval.slurm
```

The SLURM script is responsible for cluster-specific details such as:

- GPU request;
- CPU request;
- memory;
- wall time;
- account/project allocation;
- module loading;
- paths to the container;
- paths to datasets.

The Python code and experiment configuration should remain the same.

Conceptually:
```
                  Git repository
                        |
                        v
                same Python code
                        |
                        v
                 same config
                        |
              +---------+---------+
              |                   |
           Rorqual              Narval
              |                   |
       Rorqual SLURM       Narval SLURM
              |                   |
              +---------+---------+
                        |
                        v
                same .sif image
                        |
                        v
                 same experiment
```
</details>


<details>
20. Example SLURM script

A cluster-specific script might look conceptually like:
```
#!/bin/bash
#SBATCH --job-name=odir-v1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

module load apptainer

CONTAINER=/project/<account>/containers/retinal-ablation-1.0.sif
DATA_DIR=/project/<account>/datasets/ODIR

apptainer exec --nv \
    "$CONTAINER" \
    python src/train.py \
    --config configs/variant1.yaml \
    --data-dir "$DATA_DIR"
```

The exact ```#SBATCH``` directives must be adapted to the target cluster.

Do not copy Rorqual-specific resource directives directly to Narval.

21. Reproducibility requirements

Every published experiment should record:

- Git commit/tag 
- Container version
- Experiment configuration
- Dataset version/location
- Random seed
- GPU type
- Training parameters


For example:
```
Project release:       v1.0
Git commit:            abc1234
Container:             retinal-ablation-1.0.sif
Experiment:             variant1.yaml
Random seed:            42
Dataset:                ODIR version X
```

The output directory should contain these details whenever practical.

22. Random seeds

Training should use an explicitly recorded random seed.

For example:
```
python src/train.py \
    --config configs/variant1.yaml \
    --seed 42
```

For stronger reproducibility, consider running multiple seeds:
```
seed 42
seed 43
seed 44
seed 45
seed 46
```

and reporting the mean and variation rather than relying on a single training run.

23. Container versus virtual environment

The virtual environment and container have different purposes.

Virtual environment

Use ```.venv``` for:

- development;
- debugging;
- installing packages;
- writing and testing code;
- rapid iteration.
- Container

Use Apptainer for:

- reproducible experiments;
- publication;
- sharing with other researchers;
- running the same software stack on different HPC systems.

The container is the canonical execution environment for published experiments.

24. What should and should not change between clusters?

- Should remain unchanged
    ```
    src/
    configs/
    requirements.txt
    Dockerfile
    experiment definitions
    model implementation
    training implementation
    evaluation implementation
    ```

- May change:
    ```
    SLURM resource requests
    module commands
    dataset path
    container location
    project/account path
    scratch directory
    GPU allocation
    ```

The goal is:

Change the cluster configuration, not the experiment itself.

</details>

<details>
  
<summary>25. Troubleshooting</summary>



```apptainer: command not found```

Load the appropriate Apptainer module:
```
module avail
```

and consult the target cluster's documentation.
```
torch.cuda.is_available() returns False
```
Check:
- You requested a GPU.
- The job actually received a GPU.
- You used apptainer exec --nv.
- The cluster's NVIDIA driver is compatible with the container.
- The appropriate cluster GPU environment is loaded.
- Dataset cannot be found

Check:

```ls "$DATA_DIR"```


and verify that the path supplied to the experiment is correct.

Do not modify the source code merely to hard-code a new dataset path.

Container works on one cluster but not another

Check:
- GPU type
- NVIDIA driver
- Apptainer version
- container CUDA/PyTorch compatibility


The container isolates the software environment, but it still relies on the host HPC system for the kernel, GPU, and NVIDIA driver.

26. Final reproduction checklist

A new researcher should be able to follow this checklist:

[ ] Clone repository
[ ] Checkout experiment release
[ ] Obtain matching .sif container
[ ] Load Apptainer
[ ] Obtain/configure dataset
[ ] Request GPU
[ ] Test CUDA inside container
[ ] Run small sanity check
[ ] Submit experiment with SLURM
[ ] Save results
[ ] Record Git commit
[ ] Record container version
[ ] Record configuration
[ ] Record random seed


If these steps work, the experiment should be reproducible on another compatible HPC cluster without recreating the original Python environment.

27. Recommended release strategy

For each major experimental release, maintain a matching set:
```
Git tag
    |
    +-- source code
    +-- configs
    +-- Dockerfile
    +-- requirements.txt
    |
    +-- Container
         |
         +-- retinal-ablation-1.0.sif
```

For example:
```
v1.0
  └── retinal-ablation-1.0.sif

v2.0
  └── retinal-ablation-2.0.sif
```

Never silently replace the container associated with an existing published result.

This allows another researcher to reproduce an old experiment even after the project has evolved.

28. Recommended philosophy

The repository should answer three questions clearly:

What code was used?

Git provides this.

What software environment was used?

The container provides this.

What experiment was run?

The configuration file and recorded metadata provide this.

Together:
```
Git
 +
Container
 +
Configuration
 +
Dataset version
 +
Seed
 =
Reproducible experiment
```



</details>
