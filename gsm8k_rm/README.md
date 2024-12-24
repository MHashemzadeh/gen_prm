# Generative Process Reward Model

## Here is the code for GSM8K dataset with llama 3.1-8B-Instruct

### Installation

To install the packages, first create a virtual environment and then follow these steps, which are worked for the Mila cluster:

```sh

module load python/3.10
module load cuda/12.1.1
python -m venv envs/veri_env
pip install -r req.txt 
MAX_JOBS=2 pip install flash-attn --no-build-isolation

```

Activate venv:

```sh
module load python/3.10
module load cuda/12.1.1
source ../envs/veri_env/bin/activate 

```

Models ids:

- "Llama 3.1-8B-Instruct",

---

### Execute Code


`train_direct_prm.py` is used to train the generative reward model, while `train_cot_prm.py` trains the model using CoT.

To execute these scripts, use the following commands:

```sh
python train_direct_prm.py <config_direct.yaml>
python train_cot_prm.py <config_cot.yaml>
```

The bash file `run.sh` contains the settings required to execute it on the Mila cluster.

---

To evaluate the GenPRM, execute `evaluation/best_of_n_with_prm.py`. For assessing iterative correctness, run `best_of_n_with_prm_iterative.py`






