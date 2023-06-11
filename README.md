## ViT image classifier

Toy project to play with Vision Transformer architecture described in https://arxiv.org/abs/2010.11929 using PyTorch-Lightning


### Install
```
!pip3 install pytorch-lightning torch torchvision matplotlib opencv-python
```


### Train
`python3 train.py --epochs=50`
![Alt text](attachments/loss_accuracy.png?raw=true "Scalars")
![Alt text](attachments/images.png?raw=true "Data")
![Alt text](attachments/attention.png?raw=true "Attention")


### Evaluate
Test inference:
`python3 test_inference.py --help`
Run a model on a stream:
`python3 run.py --help`
