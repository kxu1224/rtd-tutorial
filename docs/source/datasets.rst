Example Datasets
=====

We utilized miRNA-seq and RNA-seq datasets from four TCGA studies: Breast Invasive Carcinoma
(BRCA), Prostate Adenocarcinoma (PRAD), Skin Cutaneous Melanoma (SKCM), and Acute Myeloid
Leukemia (LAML).

Within this package, there are two main folders of datasets. One set includes the datasets that can be used as a case study.
These are the BRCASubtype datasets. Another dataset is the SKCMPositive_4 dataset which is the original SKCM that underwent marker filtering with 4 as the mean threshold.

The example datasets provided in the package can be used for a few examples which can be found in :doc:`methods`. 
SKCMPositive_4 can be used for a pilot experiment using VAE with loss ratio 1-10.
A case study can be done on the dataset BRCASubtype using WGAN-GP.
Transfer learning as an example can be done using MAF from PRAD dataset to BRCA