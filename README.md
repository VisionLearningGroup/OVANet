## [OVANet: One-vs-All Network for Universal Domain Adaptation](https://arxiv.org/pdf/2104.03344.pdf)

![OVANet Overview](images/animation_ovanet.gif)


This repository provides code for the paper.
Please go to our project page to quickly understand the content of the paper or read our paper.
### [Project Page](https://cs-people.bu.edu/keisaito/research/OVANet.html)|  [Paper](https://arxiv.org/pdf/2104.03344.pdf)

### Environment
Python 3.6.9, Pytorch 1.6.0, Torch Vision 0.7.0, [Apex](https://github.com/NVIDIA/apex).
 We used the nvidia apex library for memory efficient high-speed training.

## Data Preparation

#### Datasets

[Office Dataset](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/),
[OfficeHome Dataset](http://hemanthdv.org/OfficeHome-Dataset/), [VisDA](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification), [DomainNet](http://ai.bu.edu/M3SDA/), [NaBird](https://dl.allaboutbirds.org/nabirds)

Prepare dataset in data directory.
```
./data/amazon/images/ ## Office
./data/Real ## OfficeHome
./data/visda_train ## VisDA synthetic images
./data/visda_val ## VisDA real images
./data/dclipart ## DomainNet # We add 'd' for all directories of DomainNet to avoid confusion with OfficeHome.
./data/nabird/images ## Nabird
```

#### File splits

[File lists (txt files)](https://drive.google.com/file/d/1j_PT-gRWQQNkbwcWBuNc01D7QtCBYomN/view?usp=sharing)

File list need to be stored in ./txt, e.g.,

```
./txt/source_amazon_opda.txt ## Office
./txt/source_dreal_univ.txt ## DomainNet
./txt/source_Real_univ.txt ## OfficeHome
./txt/nabird_source.txt ## Nabird
.
.
.
```


## Training and Evaluation

All training scripts are stored in script directory.

Ex. Open Set Domain Adaptation on Office.
```
sh scripts/run_office_obda.sh $gpu-id train.py
```

### Reference
This repository is contributed by [Kuniaki Saito](http://cs-people.bu.edu/keisaito/).
If you consider using this code or its derivatives, please consider citing:

```
@article{saito2021ovanet,
  title={OVANet: One-vs-All Network for Universal Domain Adaptation},
  author={Saito, Kuniaki and Saenko, Kate},
  journal={arXiv preprint arXiv:2104.03344},
  year={2021}
}
```
