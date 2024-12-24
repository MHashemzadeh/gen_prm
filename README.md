# Generative Process Reward Model

## Here is the code for Generatove Process Reward model, GenPRM, using GSM8K and prm800k datasets 




![n](GPV.pdf "View of Example")
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

- "google/gemma-2-9b-it",
- "google/gemma-2b-it",
- LORA 

---

### Execute Code

First, in `create_prm800k_data.py`, we select relevant columns and features to create the train and test datasets.


Then, by executing `main.py`, you can train the reward model. 

You can modify the configurations in `config.yaml`, such as the model ID, number of epochs, paths for train and test datasets, LoRA and training parameters, among others.


`run_lora.sh` is the bash script used to execute the code on the Mila cluster, where parameters such as LoRA ranking are modified. We have also created `executable.sh`, a bash script that adjusts various parameters in a loop and executes `run_lora.sh` for hyper-parameter tuning.


---

<!-- Note: We first worked on prm800k with LORA but we did not get good results. Then after talking with the mentor, he suggested to use full-fine tuning and instead of prm800k use gsm8k. So then we created the code for that one as well. Here we put this one to show that we also worked on this data. All the codes are written by mysel.  -->

Note: Initially, we worked with the PRM800K dataset using LoRA, but the results were not satisfactory. After consulting with my mentor, we decided to switch to full fine-tuning and to use the GSM8K dataset instead. Consequently, I developed new code for that dataset. This note is included to acknowledge our preliminary efforts with PRM800K. All the code was written by myself.


