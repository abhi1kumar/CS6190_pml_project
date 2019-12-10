Repo for the CS6190 PML Course Project 

### Requirements
1. Python 3.6
2. [Pytorch](http://pytorch.org)
3. Cuda 10.0

#### Pip install
```bash
virtualenv --python=/usr/bin/python3.6 py36
source py36/bin/activate
pip install torch torchvision
```

### Cloning the repo
Type the following command in terminal
```bash
git clone git@github.com:abhi1kumar/pml.git
```

### Extra directories
We need to make some extra directories to store data and models
```bash
mkdir data
mkdir model
```

### Dataset
Download the following datasets and move it to ```data``` folder
1. [Bibtex](https://drive.google.com/open?id=0B3lPMIHmG6vGcy1xM2pJZ09MMGM)


### References:
Please refer the following papers:
```
@inproceedings{gaure2017probabilistic,
  title={A probabilistic framework for zero-shot multi-label learning},
  author={Gaure, Abhilash and Gupta, Aishwarya and Verma, Vinay Kumar and Rai, Piyush},
  booktitle={UAI},
  year={2017}
}
```
