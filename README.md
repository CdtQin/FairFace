# FairFace

Here is the instruction for these codes.

model.py is the main training file that includes the multi-branch data training, the CNN architecture, which is modified from ResNet101, and the loss functions.

We applied several large private datasets that can provide a lot of subjects within underrepresented groups.

These datasets are not only used for classification but also be used to estimate the distributions. We add constrains, namely kl_loss, entropy_loss and order_loss, on these estimated distributions. Here is a paper reference: https://arxiv.org/abs/2002.03662

Other files are used for feature extraction and prediction. 

For inference:

First, you should crop image. Here we provide a file (record.txt) that contain both five landmarks and bounding boxes. You can use file affine_crop.py to crop face out. Each cropped image will be generated at the same position as the original one. If a raw image has name 1.jpg, the cropped image will be named 1_croppped.jpg

Second, you should extract feature with the by this command: sh auto_run_extract_expr_flip.sh . BTW, you should modify the sh file to assign pathes before execution.

Third, three feature files will be generated after the second step. They should be test.fea.noFlip, test.fea.Flip, and test.fea.mean

Finally, you can generate prediction via generate_result.py
