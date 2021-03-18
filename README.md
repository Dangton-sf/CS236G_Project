# Optimizing oil field productivity with GAN

This is the final project for Stanford CS 236G: Generative Adversarial Networks, 2021 Winter.

## Descriptions

This project is a modified PyTorch implementation of [Conditional Image Synthesis With Auxiliary Classifier GANs](https://arxiv.org/abs/1610.09585) but using Wasserstein GAN loss and used to generate oil and gas production curves.

### Prerequisites
* [PyTorch](http://pytorch.org/) 
* [SciPy](http://www.scipy.org/install.html) 
* [NumPy](http://www.numpy.org/) 
* [Panda](http://pandas.pydata.org/) 
* [scikit-learn](http://scikit-learn.org/)
* [haste-pytorch](http://github.com/lmnt-com/haste)
### Usage

The Data Description and 3 generated dataset are provided in the dataset folder. Each case contains Eclipse reservoir simulator runtime files used to generate the dataset and the csv dataset in zip format. Extract the csv dataset before using as input.   

To train the model, make the neccessary parameters change to parameter box in the Training.py and run the train() function. You can also train the model directly by running the Training.py. 

```
###########################################################################################################
input_file = 'Producer_start_center_hetrogenous.csv'
batch_size = 50
lambda_gp = 4
lambda_l1 = 1
nb_epoch = int(1e6)
dis_iters = 4
gen_iters = 1
lr_G = 5e-5
lr_D = 4e-5
zoneout = 0.005
dropout = 0.005
test_size = 0.5
###########################################################################################################
```

To run evaluation on a trained generator, run the eval_gene(gene) function where gene is the trained generator. It will generate all 900 possible well placement cases.
A trained generator and discriminator on the Producer_start_center_hetrogenous dataset is provided. To generate the data from the provided generator, simply run Eval_center_hetrogenous.py.

## Authors

* **Dang Ton** 
* **Jingru Chen** 


