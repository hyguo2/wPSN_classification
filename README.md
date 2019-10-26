# Weighted graphlets and deep neural networks for protein structure classification
This repository contains the implementation of the method in paper *Weighted graphlets and deep neural networks for protein structure classification*.

##Getting Started

###Prerequisites
- R 3.5.1
- tensorflow 1.8

##Installing
To install the code, simply clone or download the whole repository.

```
$ git clone https://github.com/hyguo2/wPSN_classification.git
```

##Running the code
Use wPSN_Classification.py to run the classfication by specifying the dataset name in its -fl option. The following example runs the code on the dataset cath-3.30.390 used in the paper.

```
$python wPSN_Classification.py -fl cath-3.30.390
```

This example takes 5 hours to finish. The code used here is for single thread computer which is provided to meet the submission requirement. The results in the paper is based on a paralleled version with 50 CPU cores and 5 GPU cards.
