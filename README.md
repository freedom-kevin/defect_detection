## Model Description

The defect_segmentation is built on a `fully convolutional neural network` variant of the [VGG-16] net with several modifications: first, it exploits `atrous` (dilated) convolutions to increase the field-of-view; second, the number of filters in the last layers is reduced from 4096 to 1024 in order to decrease the memory consumption and the time spent on performing one forward-backward pass; third, it omits the last pooling layers to keep the whole downsampling ratio of 8.
## Requirements

TensorFlow needs to be installed before running the scripts. TensorFlow>=0.11 is supported.
## Caffe to TensorFlow conversion

To imitate the structure of the model, we have used pre-trained `.caffemodel` file.The .util/extract_params.py script saves the structure of the network, i.e. the name of the parameters with their corresponding shapes (in TF format), as well as the weights of those parameters (again, in TF format). These weights can be used to initialize the variables in the model; otherwise, the filters will be initialised using the Xavier initialisation scheme, and biases will be initiliased as 0s. To use this script you will need to install Caffe.
## Training

We initialised the network from the .caffemodel file provided by the authors. In that model, the last classification layer is randomly initialised using the Xavier scheme with biases set to zeros. The loss function is the pixel-wise softmax loss, and it is optimised using Adam. 

To train the model use the following command:
```bash
python train.py
```
## Inference
To perform inference over your own images, use the following command:
```bash
python inference.py /path/to/your/image /path/to/ckpt/file
```
