# HyperFormer: Enhancing Entity and Relation Interaction for Hyper-Relational Knowledge Graph Completion
#### This repo provides the source code & data of our paper: [HyperFormer: Enhancing Entity and Relation Interaction for Hyper-Relational Knowledge Graph Completion (CIKM2023)](https://arxiv.org/pdf/2308.06512.pdf).
## Dependencies
* conda create -n hyperformer python=3.7 -y
* PyTorch 1.8.1
* contiguous_params 1.0.0
* scipy 1.7.3
* tqdm 4.64.1
* fastmoe 0.2.0
  * download the [fastmoe](https://github.com/laekov/fastmoe) project
  * cd fastmoe folder
  * conda install "gxx_linux-64<=10" nccl -c conda-forge -y 
  * pip install -e .
* If you have problems using MoE, you can directly download the one I used [fastmoe](https://drive.google.com/file/d/1c3ijOe5PacVWyfmD2amUk0dpTbsjIHIS/view?usp=sharing).
## Running the code
### Dataset
* Download the datasets from [Here](https://drive.google.com/drive/folders/1FBopRFRe7NS75w3NvzKTW_QyqvgKRyqC?usp=drive_link).
* Create the root directory ./dataset and put the datasets in.
* You will get four types of datasets:
  * Mixed-percentage Mixed-qualifier: WD50K, JF17K, and Wikipeople;
  * Fixed-percentage Mixed-qualifier: WD50K_33, WD50K_66, WD50K_100, same as JF17K and Wikipeople.
  * Fixed-percentage Fixed-qualifier: WikiPeople-3, WikiPeople-4, same as JF17K.
  * Entities with Low Degree: WD50K_100_1_degree, WD50K_100_2_degree, WD50K_100_3_degree, WD50K_100_4_degree, same as JF17K and Wikipeople.

### Training model
Taking the WD50K dataset as an example, you can run the following script：
```python
sh run.sh
```
For other datasets, you only need to modify the following parameters, we used the same other parameters on all datasets：
* export LOG_PATH = your log path
* export SAVE_DIR_NAME = your save path
* export DATASET = the dataset you use
* export CUDA = the gpu id
* **Notes**: If you want to reproduce the results in Table 1, you need to set **--train_mode with_valid**, because all baselines use the validation set in the training process.

## Notes
* When executed conda install "gxx_linux-64<=10" nccl -c conda-forge -y, if you meet the **WARNING conda.core.envs_manager:register_env(50): Unable to register environment. Path not writable or missing.** You should modify write permission to anaconda，e.g., **sudo chown -R hzw /home/amax/anaconda3/**, *hzw* is your username, */home/amax/anaconda3/* is anaconda path. You need see: All requested packages already installed.

## Citation
If you find this code useful, please consider citing the following paper.
```
@article{
  author={Zhiwei Hu and Víctor Gutiérrez-Basulto and Zhiliang Xiang and and Ru Li and Jeff Z. Pan},
  title={HyperFormer: Enhancing Entity and Relation Interaction for Hyper-Relational Knowledge Graph Completion},
  publisher="32nd ACM International Conference on Information and Knowledge Management",
  year={2023}
}
```

## Acknowledgement
We refer to the code of [CoLE](https://github.com/nju-websoft/CoLE). Thanks for their contributions.
