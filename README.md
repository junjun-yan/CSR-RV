# CSR&RV

An official source code for paper [CSR&RV: An Efficient Value Compression Format for Sparse Matrix-vector Multiplication](https://dl.acm.org/doi/abs/10.1007/978-3-031-21395-3_5), accepted by the 19th IFIP International Conference on Network and Parallel Computing, NPC 2022. Any communications or issues are welcomed. Please contact shuaicaijunjun@126.com. If you find this repository useful to your research or work, it is really appreciate to star this repository. :heart:

-------------

### Overview

<p align = "justify"> 
    Sparse Matrix-Vector Multiplication (SpMV) plays a critical role in many areas of science and engineering applications. The storage space of value array in general real sparse matrices accounts for costly. However, the existing compressed formats cannot balance the compressed rate and computational speed. To address this issue, we propose an efficient value compression format implemented by AVX512 instructions called Compressed Sparse Row and Repetition Value (CSR&RV). This format stores each different value once and uses the indexes array to store the position of values, which reduces the storage space by compressing the value array. We conduct a series of experiments on an Intel Xeon processor and compare it with five other formats in 30 real-world matrices. Experimental results show that CSR&RV can achieve a speedup up to 3.86× (1.66× on average) and a speedup up to 12.42× (3.12× on average) for single-core and multi-core throughput, respectively. Meanwhile, our format can reduce the memory space by 48.57% on average.
</p>

<div  align="center">    
    <img src="./pic/CSRRV.pdf" width=60%/>
</div>

<div  align="center">    
    The CSR&RV format.
</div>


### Requirements

The proposed DCRN is implemented with python 3.8.5 on a NVIDIA 3090 GPU. 

Python package information is summarized in **requirements.txt**:

- torch==1.8.0
- tqdm==4.50.2
- numpy==1.19.2
- munkres==1.1.4
- scikit_learn==1.0.1

### Pre-training
We release the pre-training code.

- Google Drive: [Link](https://drive.google.com/file/d/1XRlu3Ahgwin52jluqFu2aBW6wjCwjY4M/view?usp=sharing)
- Nut store: [Link](https://www.jianguoyun.com/p/DXCOQEYQwdaSChiEjrsEIAA)

### Quick Start

- Step1: use the **dblp.zip** file or download other datasets from [Awesome Deep Graph Clustering/Benchmark Datasets](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering#benchmark-datasets) 

- Step2: unzip the dataset into the **./dataset** folder

- Step3: run 

  ```
  python main.py --name dblp --seed 3 --alpha_value 0.2 --lambda_value 10 --gamma_value 1e3 --lr 1e-4
  ```

Parameter setting

- name: the name of dataset
- seed: the random seed. 10 runs under different random seeds.
- alpha_value: the teleport probability in graph diffusion
  - PUBMED: 0.1
  - DBLP, CITE, ACM, AMAP, CORAFULL: 0.2
- lambda_value: the coefficient of clustering guidance loss.
  - all datasets: 10
- gamma_value: the coefficient of propagation regularization
  - all datasets: 1e3
- lr: learning rate
  - DBLP 1e-4
  - ACM: 5e-5
  - AMAP: 1e-3
  - CITE, PUBMED, CORAFULL: 1e-5



Tips: Limited by the GPU memory, PUBMED and CORAFULL might be out of memory during training. Thus, we adpot batch training on PUBMED and CORAFULL dataseets and the batch size is set to 2000. Please use the batch training version of DCRN [here](https://drive.google.com/file/d/185GLObsQQL3Y-dQ2aIin5YrXuA-dgpnU/view?usp=sharing).



### Results

<div  align="center">    
    <img src="./assets/result.png" width=100%/>
</div>



<div  align="center">    
    <img src="./assets/t-sne.png" width=100%/>
</div>


### Citation

If you use code or datasets in this repository for your research, please cite our paper.

```
@inproceedings{CSRRV,
    author = {Yan, Junjun and Chen, Xinhai and Liu, Jie},
    title = {CSR\&RV: An Efficient Value Compression Format for Sparse Matrix-Vector Multiplication},
    year = {2022},
    isbn = {978-3-031-21394-6},
    publisher = {Springer-Verlag},
    address = {Berlin, Heidelberg},
    doi = {10.1007/978-3-031-21395-3_5},
    booktitle = {Network and Parallel Computing: 19th IFIP WG 10.3 International Conference, NPC 2022},
    pages = {54–60},
    numpages = {7},
    location = {Jinan, China}
}
``
