<div align="center">

# HYPE: Hyperbolic Entailment Filtering for Underspecified Images and Texts

**[Wonjae Kim](https://wonjae.kim), [Sanghyuk Chun](https://sanghyukchun.github.io/home/), [Taekyung Kim](https://scholar.google.com/citations?user=u-9bdkwAAAAJ&hl=en), [Dongyoon Han](https://sites.google.com/site/dyhan0920/), [Sangdoo Yun](https://sangdooyun.github.io/)** <br>

[NAVER AI LAB](https://naver-career.gitbook.io/en/teams/clova-cic/ai-lab)

[![Paper](https://img.shields.io/badge/Paper-arxiv-green)](https://arxiv.org/abs/2404.17507)
[![Paper](https://img.shields.io/badge/Paper-ECCV_2024-blue)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5671_ECCV_2024_paper.php)

![teaser](teaser.png)
</div>

Official PyTorch implementation of "HYPE: Hyperbolic Entailment Filtering for Underspecified Images and Texts" | [arxiv](https://arxiv.org/abs/2404.17507), [ECCV](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/5671_ECCV_2024_paper.php)

### Abstract

In an era where the volume of data drives the effectiveness of self-supervised learning, the specificity and clarity of data semantics play a crucial role in model training. Addressing this, we introduce HYPerbolic Entailment filtering (HYPE), a novel methodology designed to meticulously extract modality-wise meaningful and well-aligned data from extensive, noisy image-text pair datasets. Our approach leverages hyperbolic embeddings and the concept of entailment cones to evaluate and filter out samples with meaningless or underspecified semantics, focusing on enhancing the specificity of each data sample. HYPE not only demonstrates a significant improvement in filtering efficiency but also sets a new state-of-the-art in the DataComp benchmark when combined with existing filtering techniques. This breakthrough showcases the potential of HYPE to refine the data selection process, thereby contributing to the development of more accurate and efficient self-supervised learning models. Additionally, the image specificity ϵi can be independently applied to induce an image-only dataset from an image-text or image-only data pool for training image-only self-supervised models and showed superior performance when compared to the dataset induced by CLIP score.


## Updates

- **October 2024**: Released inference code and model weights
- **Jul 16, 2024**: Published paper on arXiv

## Prerequisites

Download the following files to the project root: [hyperbolic CLIP weights](https://drive.google.com/file/d/1VF2g6m0tlHgzYzcMEYncchHXjhw-h5qo/view?usp=share_link) and [reference set](https://drive.google.com/file/d/1pdiFdZzcqoQ1nRtlHIpP0nu-BFRbfuYe/view?usp=share_link).

- `model.py` : Implementation of Hyperbolic CLIP, which is almost identical to [MERU](https://arxiv.org/abs/2304.09172) but in [OpenCLIP]((https://github.com/mlfoundations/open_clip)) style.
- `tokenizer.py` : Tokenizer copied from [https://github.com/openai/CLIP](https://github.com/openai/CLIP)
- `hyperbolic.py` : Implementation of hyperbolic space operations.
- `hyper_demo.ipynb` : Pedagogical code to show how to calculate negative Lorentizian distance (similarity) and specificity shown in the paper.

## How to run

This repository includes functionality to calculate modality-specificities and the negative Lorentzian distance only. Please refer to the [DataComp](https://github.com/mlfoundations/datacomp) repository to calculate the ImageNet clustering score and CLIP similarity for the complete composite HYPE score. However, as shown in Table 3 of HYPE paper, using only the specificities and negative Lorentzian distance is sufficient to achieve state-of-the-art results.

## How to cite

```
@inproceedings{kim2024hype,
    title={HYPE: Hyperbolic Entailment Filtering for Underspecified Images and Texts},
    author={Kim, Wonjae and Chun, Sanghyuk and Kim, Taekyung and Han, Dongyoon and Yun, Sangdoo},
    year={2024},
    booktitle={European Conference on Computer Vision (ECCV)},
}
```

## License
```
HYPE
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/) 
```
