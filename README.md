# Fast Graph Generation via Spectral Diffusion

Code for the paper [Fast Graph Generation via Spectral Diffusion](https://ieeexplore.ieee.org/abstract/document/10366850) (IEEE TPAMI 2023).

## Dependencies

GSDM is built in **Python 3.8.0** and **Pytorch 1.10.1**. Please use the following command to install the requirements:

```sh
pip install -r requirements.txt
```

## Running Experiments


### Train model

```sh
CUDA_VISIBLE_DEVICES=${gpu_ids} python main.py --type train --config ${train_config}
```

for example, 

```sh
CUDA_VISIBLE_DEVICES=0 python main.py --type train --config community_small
```

### Evaluation

For the evaluation of generic graph generation tasks, run the following command to compile the ORCA program (see http://www.biolab.si/supp/orca/orca.html):

```sh
cd evaluation/orca 
g++ -O2 -std=c++11 -o orca orca.cpp
```

To generate graphs using the trained score models, run the following command.

```sh
CUDA_VISIBLE_DEVICES=${gpu_ids} python main.py --type sample --config community_small
```


## Citation

If you found the provided code with our paper useful in your work, we kindly request that you cite our work.

```BibTex
@article{luo2023fast,
  title={Fast graph generation via spectral diffusion},
  author={Luo, Tianze and Mo, Zhanfeng and Pan, Sinno Jialin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2023},
  publisher={IEEE}
  url={https://ieeexplore.ieee.org/abstract/document/10366850}
}
```