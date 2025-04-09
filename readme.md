# Code for Learning Visual Semantic Subspace Representations (AISTATS 2025)

Paper on [OpenReview](https://openreview.net/forum?id=R3O1mD9lyZ&referrer=%5Bthe%20profile%20of%20Gabriel%20Moreira%5D(%2Fprofile%3Fid%3D~Gabriel_Moreira1))


* Train on MNIST with ConvNet
> python ./train.py --config-name=base general.name=mnist_convnet model.dim_out=10 model.dim_in=1 train.dataset=MNIST val.dataset=MNIST train.loss_coefs.alpha=0.997 train.scheduler.step_size=20 model.backbone=convnet train.max_epochs=150

* Train on MNIST with Resnet-18
> python ./train.py --config-name=base general.name=mnist_resnet18 model.dim_out=10 model.dim_in=1 train.dataset=MNIST val.dataset=MNIST train.loss_coefs.alpha=0.997 train.scheduler.step_size=20 model.backbone=resnet18 train.max_epochs=150

* Train on FashionMNIST with ConvNet
> python ./train.py --config-name=base general.name=fashionmnist_convnet model.dim_out=10 model.dim_in=1 train.dataset=FashionMNIST val.dataset=FashionMNIST train.loss_coefs.alpha=0.997 train.scheduler.step_size=30 train.optimizer.momentum=0.99 model.backbone=convnet train.max_epochs=150

* Train on FashionMNIST with Resnet-18
> python ./train.py --config-name=base general.name=fashionmnist_resnet18 model.dim_out=10 model.dim_in=1 train.dataset=FashionMNIST val.dataset=FashionMNIST train.loss_coefs.alpha=0.997 train.scheduler.step_size=30 train.optimizer.momentum=0.99 model.backbone=resnet18 train.max_epochs=150

* Train on CIFAR-10
> python ./train.py --config-name=base general.name=cifar-10 model.dim_in=3 model.dim_out=10 train.dataset=CIFAR10 val.dataset=CIFAR10 train.loss_coefs.alpha=0.999 train.loss_coefs.beta=0.01 train.scheduler.step_size=100 train.scheduler.gamma=0.1 train.max_epochs=200

* Train on CIFAR-100
> python ./train.py --config-name=base general.name=cifar-100 model.dim_in=3 model.dim_out=100 train.dataset=CIFAR10 val.dataset=CIFAR10 train.loss_coefs.alpha=0.999 train.loss_coefs.beta=0.01 train.scheduler.step_size=100 train.scheduler.gamma=0.7 train.max_epochs=200

* Celeb-A training
> python ./train.py --config-name=base general.name=celeba model.dim=25 train.dataset=CELEBA val.dataset=CELEBA train.loss_coefs.alpha=0.999 train.loss_coefs.beta=0.01 train.scheduler.step_size=100 train.scheduler.gamma=0.5
