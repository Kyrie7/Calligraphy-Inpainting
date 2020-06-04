# Calligraphy-Inpainting
 calligraphy rubbings impainting

 tensorflow -2.1.0
 numpy 1.18.1
 matplotlib 3.1.3
 tqdm 4.42.1

 
 argparse: 
 --batch-size : training batch size
 --val-batch-size : how many same characters you use in val dir
 --model-index : 1 2 3 4 is different model in mmodels
 --weight-decay : model weight decay (undone)
 --learning-rate : initial learning rate, it will decrease by 0.1 when epochs come to 40% and 60%
 --l1-lambda : the l1 loss lambda 
 --norm-type : we can use instancenorm or batchnorm
 --epochs : training epochs
 
