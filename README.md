# FPRF (AAAI 2024)
### [Project Page](https://kim-geonu.github.io/FPRF/) | [Paper](https://kim-geonu.github.io/FPRF/static/pdf/FPRF.pdf)
This repository contains a pytorch implementation for the AAAI 2024 paper, [FPRF: Feed-Forward Photorealistic Style Transfer of Large-Scale 3D Neural Radiance Fields](https://arxiv.org/abs/2401.05516).



https://github.com/Kim-GeonU/FPRF/assets/32728514/f8cb709c-16e2-49bf-aed8-75f080a5d403





## Getting Started
This code was developed on Ubuntu 18.04 with Python 3.9, CUDA 11.8 and PyTorch 2.0.0. Later versions should work, but have not been tested.

### System Requirements
- Python 3.9
- CUDA 11.8
- Single GPU w/ minimum 24 GB RAM

### Environment Setup
Create and activate a virtual environment, then install `pytorch` and `tiny-cuda-nn`:
```
conda create -n FPRF python=3.9
conda activate FPRF
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

Install the remaining requirements with pip:
```
pip install -r requirements.txt
```

### Download the LLFF Datasets
To run FPRF, please download the [LLFF dataset](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1) and put it in ./data. 

## Training Scene
Run below commands to train a stylizable 3D scene. 
```bash
PYTHONPATH=. python plenoxels/main.py --config-path plenoxels/configs/final/LLFF/llff_flower.py
```
You can change the scene by editing config files in ./plenoxels/configs/final/LLFF/.

## Style transfer 
Run below commands to transfer the style of a 3D scene to the refernece images in ./referneces.

```bash
PYTHONPATH=. python plenoxels/main.py --config-path plenoxels/configs/final/LLFF/llff_flower.py --log-dir logs/flower --render-only
```
<details><summary>Here are the controllable hyperparameters.</summary>
      
```bash
PYTHONPATH=. python plenoxels/main.py --config-path plenoxels/configs/final/LLFF/llff_flower.py --log-dir logs/flower --render-only  --style_path ./references --num_clusters 10 --local_global_blending_ratio 0.3 --temperature 100
```

* num_clusters - Number of clusters for clustering each reference image.

* local_global_blending_ratio - Ratio of global style feature for style transfer. 1 refers using only global style features and 0 refers using only local style features.

* temperature - Temperature of softmax operation for semantic correspondence matching. 

</details>

## Checkpoints
To inference with a checkpoint, please download a model.pth file from [this link](https://drive.google.com/drive/folders/1wRsHQlqbynXiKeqO81GKuaSX57wyurFF?usp=sharing) and put it in ./logs/{scene}, _e.g._ ./logs/flower/model.pth .

## Citation
If you find our code or paper helps, please consider citing:
````BibTeX
@misc{kim2024fprf,
      title={FPRF: Feed-Forward Photorealistic Style Transfer of Large-Scale 3D Neural Radiance Fields}, 
      author={GeonU Kim and Kim Youwang and Tae-Hyun Oh},
      year={2024},
      eprint={2401.05516},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
````

## Acknowledgement
This work was supported by the LG Display (2022008004), Institute of Information \& communications Technology Planning \& Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2022-0-00290,Visual Intelligence for Space-Time Understanding and Generation based on Multi-layered Visual Common Sense; No.RS-2023-00225630, Development of Artificial Intelligence for Text-based 3D Movie Generation; No.2021-0-02068, Artificial Intelligence Innovation Hub; No. 2019-0-01906, Artificial Intelligence Graduate School Program(POSTECH))

The implementation of FPRF is largely inspired and fine-tuned from the seminal prior work, [K-Planes](https://github.com/sarafridov/K-Planes) (Fridovich-Keil _et al._).
We thank the authors of K-Planes who made their code public.
