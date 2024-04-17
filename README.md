# Power line segmentation by Multi-level attention from Hough domain

### Inference code
Code for reproducing results in the paper __Power line segmentation by Multi-level attention from Hough domain__.

## Network Architecture
![pipeline](https://github.com/JYYMALL/PLNet/blob/main/pipeline.png)

## Quantitative Performance
|       | F1 | IoU | Recall | Precision |
| :------:| :--------: | :--------: |:--------: | :--------: |
| [UNet](https://arxiv.org/pdf/1505.04597.pdf) | 0.8281 | 0.7261 |0.7963 | 0.8840 |
|[BASNet](https://arxiv.org/pdf/2101.04704.pdf)| 0.8287 | 0.7410 |0.8012 | 0.8623 |
| [SegFormer](https://proceedings.neurips.cc/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf)| 0.7796 | 0.6496 |0.8496 | 0.7322 |
| [SegNext](https://proceedings.neurips.cc/paper_files/paper/2022/file/08050f40fff41616ccfc3080e60a301a-Paper-Conference.pdf)| 0.8353 | 0.7288 |0.8451 | 0.8410 |
| [BGNet](https://arxiv.org/pdf/2207.00794.pdf) | 0.8618 | 0.7673 |0.8819 | 0.8369 |
| Ours| 0.8791 | 0.7903 |0.8958 | 0.8703 |

## Results
<p align="center">
<img src="https://github.com/JYYMALL/PLNet/blob/main/result.png", width="720">
</p>

__Note__: Visualization of segmented results on TTPLA dataset. (a) Original Images. (b) Ground truth. (c) [UNet](https://arxiv.org/pdf/1505.04597.pdf). (d) [BASNet](https://arxiv.org/pdf/2101.04704.pdf).
(e) [SegFormer](https://proceedings.neurips.cc/paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf). (f) [SegNext](https://proceedings.neurips.cc/paper_files/paper/2022/file/08050f40fff41616ccfc3080e60a301a-Paper-Conference.pdf). (g) [BGNet](https://arxiv.org/pdf/2207.00794.pdf). (h) Ours.

## Require
Please `pip install` the following packages:
- Cython
- torch>=1.5
- torchvision>=0.6.1
- matplotlib
- imageio
- numpy
- opencv
- deep-hough
  
__Note__:Please make sure to install __deep-hough__ on an Ubuntu system. For specific installation instructions, please refer to [this link](https://github.com/Hanqer/deep-hough-transform#requirements).
## Testing:
1. Configuring your environment (Prerequisites):
    
    + Creating a virtual environment in terminal: `conda create -n plnet python=3.7`.
    
    + Installing necessary packages.

2. Downloading necessary data:

    + downloading testing dataset and move it into `./data/test/`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1iOwioenpnfYKlpOXZIHWykCbRo1YLI-U/view?usp=sharing).
    
    + downloading pretrained weights and move it into `./checkpoints/PLNet/PLNet-BEST.pth`, 
    which can be found in this [download link (Google Drive)](https://drive.google.com/file/d/1jT5Fx7kbYVkoXiwAOOQP7ZTtDGoTtYOx/view?usp=sharing).
    
    + downloading Res2Net weights and move it into `./models/res2net50_v1b_26w_4s-3cf99910.pth`[download link (Google Drive)](https://drive.google.com/file/d/1EFoiK8XDzTZKjPsruHPwEtKJ65v3W9Ib/view?usp=sharing).

3. Testing Configuration:

    + After you download all the pretrained model and testing dataset, just run `test.py` to generate the final prediction map: 
    replace your trained model directory (`--pth_path`).


