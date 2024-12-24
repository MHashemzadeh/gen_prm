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




---

<!-- Note: We first worked on prm800k with LORA but we did not get good results. Then after talking with the mentor, he suggested to use full-fine tuning and instead of prm800k use gsm8k. So then we created the code for that one as well. Here we put this one to show that we also worked on this data. All the codes are written by mysel.  -->




