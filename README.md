# FairFace

Here is the instruction for these codes.

model.py is the main training file that includes the multi-branch data training, the CNN architecture, which is modified from ResNet101, and the loss functions.

We applied several large private datasets that can provide a lot of subjects within underrepresented groups.

These datasets are not only used for classification but also be used to estimate the distributions. We add constrains, namely kl_loss, entropy_loss and order_loss, on these estimated distributions. Here is a paper reference: https://arxiv.org/abs/2002.03662

Other files are used for feature extraction and prediction. We extract features for both the original image and flipped image and calculate the mean of them as the final representation.
