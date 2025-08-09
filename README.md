# Unsupervised-Domain-Adaptation-for-Automatic-Polyp-Segmentation-using-Synthetic-Data

## Abstract

Colorectal cancer is a significant health concern that can often be prevented through early detection of precancerous polyps during routine screenings. Although artificial intelligence (AI) methods have shown potential in reducing polyp miss rates, clinical adoption remains limited due to concerns over patient privacy, limited access to annotated data, and the high cost of expert labeling. To address these challenges, we propose an unsupervised domain adaptation (UDA) approach that leverages a fully synthetic colonoscopy dataset, SynthColon, and adapts it to real-world, unlabeled data. Our method builds on the DAFormer framework and integrates a Transformer-based hierarchical encoder, a context-aware feature fusion decoder, and a self-training strategy. We evaluate our approach on the Kvasir-SEG and CVC-ClinicDB datasets. Results show that our method achieves improved segmentation performance of 69% mIoU, compared to the baseline approach from the original SynthColon study and remains competitive with models trained on enhanced versions of the dataset.

## Comparison with SynthColon-based methods when Kvasir-Seg is the target domain. 


| Method       | Src-only (%) | UDA (%) | Oracle (%) |
|--------------|--------------|---------|------------|
| DAFormer     | 54.85        | 69.0    | 85.18      |
| SynthColon[^1] | 52.7         | –       | 85.70      |
| CUT-Seg[^2]   | 62.1         | –       | 85.70      |
| PL-CUT-Seg[^3] | –            | 68.77   | 85.70      |

While CUT-Seg and PL-CUT-Seg benefit from a more refined source dataset than ours, their results do not surpass the performance achieved by our approach


## Setup

* The Kvasir-SEG dataset is publically available and may be downloaded from here: 
      https://datasets.simula.no/kvasir-seg/

* Synth Colon may be downloaded here: https://enric1994.github.io/synth-colon/
  Please downloade the full size (41.2 GB) folder to obtain the realistic images.

* The SegFormer pretrained weights can be obtained [here](https://connecthkuhk-my.sharepoint.com/personal/xieenze_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fxieenze%5Fconnect%5Fhku%5Fhk%2FDocuments%2Fsegformer%2Fpretrained%5Fmodels&ga=1).

  ## References

[^1]: Moreu, E.; McGuinness, K.; O’Connor, N.E. Synthetic data for unsupervised polyp segmentation, 2022, [arXiv:2202.08680](https://arxiv.org/abs/2202.08680).

[^2]: Moreu, E.; Arazo, E.; McGuinness, K.; O’Connor, N.E. Joint one-sided synthetic unpaired image translation and segmentation for colorectal cancer prevention. *Expert Systems*, 2022, 40. [https://doi.org/10.1111/exsy.13137](https://doi.org/10.1111/exsy.13137).

[^3]: Moreu, E.; Arazo, E.; McGuinness, K.; O’Connor, N.E. Self-Supervised and Semi-Supervised Polyp Segmentation using Synthetic Data, 2023, [arXiv:2307.12033](https://arxiv.org/abs/2307.12033).


  

