# Deep Clustering for Unsupervised Learning of Visual Features

This code implements the unsupervised training of convolutional neural networks, or convnets, as described in the paper [Deep Clustering for Unsupervised Learning of Visual Features](https://arxiv.org/abs/1807.05520).

Moreover, we provide the evaluation protocol codes we used in the paper:
* Pascal VOC classification, detection and segmentation
* Linear classification on activations
* Instance-level image retrieval

Finally, this code also includes a visualisation module that allows to assess visually the quality of the learned features.

## Requirements

- a Python intallation version 2.7
- the SciPy and scikit-learn packages
- a PyTorch install ([pytorch.org](http://pytorch.org))
- a Faiss install ([Faiss](https://github.com/facebookresearch/faiss))
- Download ImageNet dataset


## Pre-trained models
We provide pre-trained models with AlexNet and VGG-16 architectures, available for download.
* The models in Caffe format expect BGR inputs that range in [0, 255]. You do not need to subtract the per-color-channel mean image since the preprocessing of the data is already included in our released models.
* The models in PyTorch format expect RGB inputs that range in [0, 1]. You should preprocessed your data before passing them to the released models by normalizing them: ```mean_rgb = [0.485, 0.456, 0.406]```; ```std_rgb = [0.229, 0.224, 0.225] ```
Note that in all our released models, sobel filters are computed within the models as two convolutional layers (greyscale + sobel filters).

You can download all variants by running 
```
$ ./download_model.sh
```
This will fetch the models into `${HOME}/deepcluster_models` by default.
You can change that path in the environment variable.
Direct download links are provided here:
* [AlexNet-PyTorch](https://dl.fbaipublicfiles.com/deepcluster/alexnet/checkpoint.pth.tar)
* [AlexNet-prototxt](https://dl.fbaipublicfiles.com/deepcluster/alexnet/model.prototxt) + [AlexNet-caffemodel](https://dl.fbaipublicfiles.com/deepcluster/alexnet/model.caffemodel)
* [VGG16-PyTorch](https://dl.fbaipublicfiles.com/deepcluster/vgg16/checkpoint.pth.tar)
* [VGG16-prototxt](https://dl.fbaipublicfiles.com/deepcluster/vgg16/model.prototxt) + [VGG16-caffemodel](https://dl.fbaipublicfiles.com/deepcluster/vgg16/model.caffemodel)

We also provide the last epoch cluster assignments for these models. After downloading, open the file with Python 2:
```
import pickle
with open("./alexnet_cluster_assignment.pickle", "rb") as f:
    b = pickle.load(f)
```
If you're a Python 3 user, specify ```encoding='latin1'``` in the load fonction.
Each file is a list of (image path, cluster_index) tuples.
* [AlexNet-clusters](https://dl.fbaipublicfiles.com/deepcluster/alexnet/alexnet_cluster_assignment.pickle)
* [VGG16-clusters](https://dl.fbaipublicfiles.com/deepcluster/vgg16/vgg16_cluster_assignment.pickle)

## Running the unsupervised training

Unsupervised training can be launched by running:
```
$ ./main.sh
```
Please provide the path to the data folder:
```
DIR=/datasets01/imagenet_full_size/061417/train
```
To train an AlexNet network, specify `ARCH=alexnet` whereas to train a VGG-16 convnet use `ARCH=vgg16`.

You can also specify where you want to save the clustering logs and checkpoints using:
```
EXP=exp
```

During training, models are saved every other n iterations (set using the `--checkpoints` flag), and can be found in for instance in `${EXP}/checkpoints/checkpoint_0.pth.tar`.
A log of the assignments in the clusters at each epoch can be found in the pickle file `${EXP}/clusters`.


Full documentation of the unsupervised training code `main.py`:
```
usage: main.py [-h] [--arch ARCH] [--sobel] [--clustering {Kmeans,PIC}]
               [--nmb_cluster NMB_CLUSTER] [--lr LR] [--wd WD]
               [--reassign REASSIGN] [--workers WORKERS] [--epochs EPOCHS]
               [--start_epoch START_EPOCH] [--batch BATCH]
               [--momentum MOMENTUM] [--resume PATH]
               [--checkpoints CHECKPOINTS] [--seed SEED] [--exp EXP]
               [--verbose]
               DIR

PyTorch Implementation of DeepCluster

positional arguments:
  DIR                   path to dataset

optional arguments:
  -h, --help            show this help message and exit
  --arch ARCH, -a ARCH  CNN architecture (default: alexnet)
  --sobel               Sobel filtering
  --clustering {Kmeans,PIC}
                        clustering algorithm (default: Kmeans)
  --nmb_cluster NMB_CLUSTER, --k NMB_CLUSTER
                        number of cluster for k-means (default: 10000)
  --lr LR               learning rate (default: 0.05)
  --wd WD               weight decay pow (default: -5)
  --reassign REASSIGN   how many epochs of training between two consecutive
                        reassignments of clusters (default: 1)
  --workers WORKERS     number of data loading workers (default: 4)
  --epochs EPOCHS       number of total epochs to run (default: 200)
  --start_epoch START_EPOCH
                        manual epoch number (useful on restarts) (default: 0)
  --batch BATCH         mini-batch size (default: 256)
  --momentum MOMENTUM   momentum (default: 0.9)
  --resume PATH         path to checkpoint (default: None)
  --checkpoints CHECKPOINTS
                        how many iterations between two checkpoints (default:
                        25000)
  --seed SEED           random seed (default: 31)
  --exp EXP             path to exp folder
  --verbose             chatty
```


## Evaluation protocols

### Pascal VOC

To run the classification task with fine-tuning launch:
```
./eval_voc_classif_all.sh
```
and with no finetuning:
```
./eval_voc_classif_fc6_8.sh
```

Both these scripts download [this code](https://github.com/philkr/voc-classification).
You need to download the [VOC 2007 dataset](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/). Then, specify in both `./eval_voc_classif_all.sh` and `./eval_voc_classif_fc6_8.sh` scripts the path `CAFFE` to point to the caffe branch, and `VOC` to point to the Pascal VOC directory.
Indicate in `PROTO` and `MODEL` respectively the path to the prototxt file of the model and the path to the model weights of the model to evaluate.
The flag `--train-from` allows to indicate the separation between the frozen and to-train layers.

We implemented [voc classification](https://github.com/facebookresearch/deepcluster/blob/master/eval_voc_classif.py) with PyTorch.

Erratum: When training the MLP only (fc6-8), the parameters of scaling of the batch-norm layers in the whole network are trained. 
With freezing these parameters we get 70.4 mAP.

### Linear classification on activations

You can run these transfer tasks using:
```
$ ./eval_linear.sh
```

You need to specify the path to the supervised data (ImageNet or Places):
```
DATA=/datasets01/imagenet_full_size/061417/
```
the path of your model:
```
MODEL=/private/home/mathilde/deepcluster/checkpoint.pth.tar
```
and on top of which convolutional layer to train the classifier:
```
CONV=3
```

You can specify where you want to save the output of this experiment (checkpoints and best models) with
```
EXP=exp
```

Full documentation for this task:
```
usage: eval_linear.py [-h] [--data DATA] [--model MODEL] [--conv {1,2,3,4,5}]
                      [--tencrops] [--exp EXP] [--workers WORKERS]
                      [--epochs EPOCHS] [--batch_size BATCH_SIZE] [--lr LR]
                      [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY]
                      [--seed SEED] [--verbose]

Train linear classifier on top of frozen convolutional layers of an AlexNet.

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           path to dataset
  --model MODEL         path to model
  --conv {1,2,3,4,5}    on top of which convolutional layer train logistic
                        regression
  --tencrops            validation accuracy averaged over 10 crops
  --exp EXP             exp folder
  --workers WORKERS     number of data loading workers (default: 4)
  --epochs EPOCHS       number of total epochs to run (default: 90)
  --batch_size BATCH_SIZE
                        mini-batch size (default: 256)
  --lr LR               learning rate
  --momentum MOMENTUM   momentum (default: 0.9)
  --weight_decay WEIGHT_DECAY, --wd WEIGHT_DECAY
                        weight decay pow (default: -4)
  --seed SEED           random seed
  --verbose             chatty
```

### Instance-level image retrieval

You can run the instance-level image retrieval transfer task using:
```
./eval_retrieval.sh
```

## Visualisaton

We provide two standard visualisation methods presented in our paper.

### Filter visualisation with gradient ascent

First, it is posible to learn an input image that maximizes the activation of a given filter. We follow the process
described by [Yosinki et al.](https://arxiv.org/abs/1506.06579) with a cross entropy function between the target
filter and the other filters in the same layer.
From the visu folder you can run
```
./gradient_ascent.sh
```
You will need to specify the model path ```MODEL```, the architecture of your model ```ARCH```, the path of the folder in which you want to save the synthetic images ```EXP``` and the convolutional layer to consider ```CONV```.

Full documentation:
```
usage: gradient_ascent.py [-h] [--model MODEL] [--arch {alexnet,vgg16}]
                          [--conv CONV] [--exp EXP] [--lr LR] [--wd WD]
                          [--sig SIG] [--step STEP] [--niter NITER]
                          [--idim IDIM]

Gradient ascent visualisation

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Model
  --arch {alexnet,vgg16}
                        arch
  --conv CONV           convolutional layer
  --exp EXP             path to res
  --lr LR               learning rate (default: 3)
  --wd WD               weight decay (default: 10^-5)
  --sig SIG             gaussian blur (default: 0.3)
  --step STEP           number of iter between gaussian blurs (default: 5)
  --niter NITER         total number of iterations (default: 1000)
  --idim IDIM           size of input image (default: 224)
```
### Top 9 maximally activated images in a dataset

Finally, we provide code to retrieve images in a dataset that maximally activate a given filter in the convnet.
From the visu folder, after having changed the fields ```MODEL```, ```EXP```, ```CONV``` and ```DATA```, run
```
./activ-retrieval.sh
```

## License

You may find out more about the license [here](https://github.com/facebookresearch/deepcluster/blob/master/LICENSE).

## Reference

If you use this code, please cite the following paper:

Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. "Deep Clustering for Unsupervised Learning of Visual Features." Proc. ECCV (2018).

```
@InProceedings{caron2018deep,
  title={Deep Clustering for Unsupervised Learning of Visual Features},
  author={Caron, Mathilde and Bojanowski, Piotr and Joulin, Armand and Douze, Matthijs},
  booktitle={European Conference on Computer Vision},
  year={2018},
}
```
