# UNET
UNET Implementation in Pytorch
To train the model use python train.py command. It is recommended to use --amp command if you have a newer generation of graphics card like Volta and RTX architecture.
Wandb will upload the results of your model to its web application where you can see the mask created by the model during training.
Once the model has been trained you can test the prediction using python predict.py -i input_image.jpg -o output_image.jpg.
-i stands for input image to be given to the model and -o for output image. Multiple images can be passed in the -i argument
