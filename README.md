# ReimplementCycleGAN
### Dataset
1. To obtain one of the CycleGAN datasets, modify the get_dataset.sh in the datasets folder to obtain desired dataset.
2. execute the following command, which will download and unzip the dataset:
```
$sh get_dataset.sh
```

### Training
After downloading and unzipping the dataset, we will need to change our directory to the cyclegan folder, and execute the Training_notebook.py file. An example command is listed below:
```
$ python Train_notebook.py --dataroot "../datasets/horse2zebra" --batchSize 1 --epochs 100 --decay_epoch 50 --lambd_identity 5
```

Every batch_id whose divides 500 equals to 0, the batch_id images will be saved in the training_output folder.

At each epoch, the model will output a log in the logs folder looks like the following:
```
TimeStamp:Tue Dec  3 15:48:57 2019,Epoch:1, Training Time: 667.3569431304932,lossD_A :0.21292290514946957,lossD_B:0.1849169862480171,loss_G:8.614637949656153
```

### Testing
In order to test all the images, execute the following command:
```
$python test.py --dataroot ../datas
ets/horse2zebra/ --model models/C_GAN_datasets_ld_10_id_5.model 
```
Test results will be saved in the test_output folder

### Dataset reference
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/datasets.md
