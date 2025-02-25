# 2DGS 
## 2D (Surfel) Gaussian Splatting Implemented in Nerfstudio


<div align="center">
<img src="media/3DGSvs2DGSrgb.gif"/>
<div style="height: 50px;">&nbsp;</div>
<img src="media/3DGSvs2DGSdepth.gif"/>
<div style="height: 50px;">&nbsp;</div>
(Left) 3D Gaussian Splatting, (Right) 2D Gaussian Splatting
</div>

Tested on Python 3.10, cuda 12.1, using conda. 

## Installation
1. Create conda environment and install relevant packages
```
conda create --name 2dgs -y python=3.10
conda activate 2dgs
conda install -c "nvidia/label/cuda-12.1.0" cuda-toolkit
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install nerfstudio==1.1.5
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

2. Install 2DGS!
```
git clone https://github.com/uynitsuj/2dgs-nerfstudio.git
cd 2dgs-nerfstudio
python -m pip install -e .

ns-install-cli
```