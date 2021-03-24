# []()

This repository provides code for the paper, .
Please go to our project page to quickly understand the content of the paper or read our paper.
### [Project Page](http://cs-people.bu.edu/keisaito/research/DANCE.html)  [Paper]()


## Environment
Python 3.6.9, Pytorch 1.2.0, Torch Vision 0.4, [Apex](https://github.com/NVIDIA/apex). See requirement.txt.
 We used the nvidia apex library for memory efficient high-speed training.

## Data Preparation

[Office Dataset](https://people.eecs.berkeley.edu/~jhoffman/domainadapt/)
[OfficeHome Dataset](http://hemanthdv.org/OfficeHome-Dataset/) [VisDA](https://github.com/VisionLearningGroup/taskcv-2017-public/tree/master/classification)

Prepare dataset in data directory.
```
./data/amazon/images/ ## Office
./data/Real ## OfficeHome
./data/visda_train ## VisDA synthetic images
./data/visda_val ## VisDA real images

```

File list is stored in ./txt.

## Train

All training script is stored in script directory.


Example of Open Set Domain Adaptation on Office.
```
sh script/run_office_obda.sh $gpu-id configs/office-train-config_ODA.yaml
```


### Reference
This repository is contributed by [Kuniaki Saito](http://cs-people.bu.edu/keisaito/).
If you consider using this code or its derivatives, please consider citing: