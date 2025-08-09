# Unsupervised-Domain-Adaptation-for-Automatic-Polyp-Segmentation-using-Synthetic-Data

## Abstract

Colorectal cancer is a significant health concern that can often be prevented through early detection of precancerous polyps during routine screenings. Although artificial intelligence (AI) methods have shown potential in reducing polyp miss rates, clinical adoption remains limited due to concerns over patient privacy, limited access to annotated data, and the high cost of expert labeling. To address these challenges, we propose an unsupervised domain adaptation (UDA) approach that leverages a fully synthetic colonoscopy dataset, SynthColon, and adapts it to real-world, unlabeled data. Our method builds on the DAFormer framework and integrates a Transformer-based hierarchical encoder, a context-aware feature fusion decoder, and a self-training strategy. We evaluate our approach on the Kvasir-SEG and CVC-ClinicDB datasets. Results show that our method achieves improved segmentation performance of 69% mIoU, compared to the baseline approach from the original SynthColon study and remains competitive with models trained on enhanced versions of the dataset.

## Setup

* The Kvasir-SEG dataset is publically available and may be downloaded from here: 
      https://datasets.simula.no/kvasir-seg/

* Synth Colon may be downloaded here: https://enric1994.github.io/synth-colon/
  Please downloade the full size (41.2 GB) folder to obtain the realistic images.

* The SegFormer pretrained weights can be obtained [here](https://connecthkuhk-my.sharepoint.com/personal/xieenze_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxieenze%5Fconnect%5Fhku%5Fhk%2FDocuments%2Fsegformer%2Fpretrained%5Fmodels&ga=1). 

  

