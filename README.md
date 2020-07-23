# FairFace

For training method:

Our main method lies in two parts, multi-branch training and similarity distribution manipulation. We conduct classification on different branches. Besides, we estimate the similarity distributions for these branches and add constrains on them. These two parts contribute to both the accuracy improvement and the alleviation of bias among different groups.

model.py is the main training file that includes the multi-branch data training, the CNN architecture, which is modified from ResNet101, and the loss functions.

We applied several large private datasets that can provide a lot of subjects within underrepresented groups.

These datasets are not only used for classification but also be used to estimate the distributions. We add constrains, namely kl_loss, entropy_loss and order_loss, on these estimated distributions. Here is a paper reference: https://arxiv.org/abs/2002.03662

To train with this model, please call the function loss() at line 556. All the arguments of this fuction are lists of tensors, an d each of them represents a branch. You can easily get what those tensors should contain from the name of arguments. For normal branches, with input argument 'images' and 'labels', we conduct a normal classification task, while for paired branches, with input argument 'paired_images' and 'paired_labels', we use kl_loss, entropy_loss and order_loss to manipulate the distribution. In our implementation, positive pairs are offline mined and input as paired branches, while negative pairs are online mined from normal branches.

For inference:

First, you should crop images. Here we provide a file (record.txt) that contain either five landmarks or bounding boxes for each image. You can use file affine_crop.py to crop face out. Each cropped image will be generated at the same position as the original one. If a raw image has name 1.jpg, the cropped image will be named 1_croppped.jpg

Second, you should extract feature with the by this command: sh auto_run_extract_expr_flip.sh . BTW, you should modify the sh file to assign pathes before execution. (model file link: https://drive.google.com/file/d/1N10oud9GKUy5azZV8NjCmdyqBlCqacYn/view?usp=sharing)

Third, three feature files will be generated after the second step. They should be test.fea.noFlip, test.fea.Flip, and test.fea.mean

Finally, you can generate prediction via generate_result.py

Enviroment:

Please try Cuda 9.1, Cudnn 7.0, and TensorFlow 1.8
