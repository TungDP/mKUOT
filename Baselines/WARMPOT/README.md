# WARMPOT

Official implementation of **WARMPOT**, proposed in *Theoretical Performance Guarantees for Partial Domain Adaptation via Partial Optimal Transport* ([arXiv:2506.02712](https://arxiv.org/abs/2506.02712)).

### Requirement
- torch==1.12.0
- [POT==0.9.5](https://github.com/PythonOT/POT) 

### Usage
#### Data perparation

1. **OfficeHome**
   1. Download the dataset from the [official website](https://www.hemanthdv.org/officeHomeDataset.html) to `data/OfficeHome`
   2. Rename the folder of ``real world``  to `Real_World` 

2. **ImageNet**: 
   1. Download the ImageNet dataset (`ILSVRC2012_img_train.tar`) from the [official website](https://www.image-net.org/download.php) to `data/ImageNetCaltech`.
   2. Uncompress the ImageNet dataset to a `train/` folder. 

#### Script

The main script is `warmpot.py`. Commands for reproducing the results in the paper are provided in `run.sh`

#### Results

There will be some randomness in the results due to random initialization. We report the results over several random seeds.

**OfficeHome**

| seed              |2020|2021|2022|2025|2026|2027|Avg       |
|-------------------|----|----|----|----|----|----|----------|
| A→C	|60.8|61.9|61.9|63.0|63.6|63.8|62.5 (1.2)| 
| A→P	|82.2|81.4|84.1|82.8|83.5|84.2|83.0 (1.1)| 
| A→R	|89.7|89.8|89.2|89.1|89.9|89.2|89.5 (0.3)| 
| C→A	|76.6|75.1|75.4|73.2|75.7|75.1|75.2 (1.1)| 
| C→P	|82.1|78.2|78.5|76.4|79.1|76.0|78.4 (2.2)| 
| C→R	|84.2|82.0|82.3|80.7|83.5|81.2|82.3 (1.3)| 
| P→A	|76.8|75.0|75.6|77.4|79.2|75.8|76.6 (1.5)| 
| P→C	|65.2|58.3|64.1|59.0|63.0|58.9|61.4 (3.1)| 
| P→R	|88.8|89.3|87.8|87.5|86.5|88.1|88.0 (1.0)| 
| R→A	|80.6|82.0|80.1|80.8|81.7|81.3|81.1 (0.7)| 
| R→C	|67.3|66.9|66.9|64.8|67.1|66.0|66.5 (1.0)| 
| R→P	|86.3|87.2|87.3|85.7|86.5|86.6|86.6 (0.6)| 
||||||||77.6 (0.7)|

**ImageNetCaltech**

|seed|2020|2021|2022| Avg|
|----|----|----|----|----|
|I→C|84.7|84.7|84.9|84.8 (0.1)|

### Reference
If you find the code useful, please cite the following papers.

        @inproceedings{naram2025theoretical,
            title={Theoretical Performance Guarantees for Partial Domain Adaptation via Partial Optimal Transport},
            author={Naram, Jayadev and Hellstr{\"o}m, Fredrik and Wang, Ziming and J{\"o}rnsten, Rebecka and Durisi, Giuseppe},
            booktitle={Proc. Int. Conf. Mach. Learning (ICML)},
            year={2025},
            address = {Vancouver, Canada},
            month = {July},
        }

For any question, welcome to open an issue or contact me.

### Acknowledgement

The `datasets` module was adopted from the [Transfer Learning Library](https://github.com/thuml/Transfer-Learning-Library). The implementation of training loop based on mini-batch partial optimal transport was adopted from [Mini-batch-OT](https://github.com/khainb/Mini-batch-OT).
We thank the authors of these repositories and other authors in the community for their code.


### LICENSE
The code is available under a [MIT license](LICENSE).
