### Official implementation for the CVPR 2024 paper CAMEL.



### Install requirements ###

```bash
pip install -r requirements.txt
```

### Download pre-trained Stable-diffusion-v1-4 ### 

```bash
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4 checkpoints/stable-diffusion-v1-4
```

### Training and Evaludation  ### 

To fine-tune the text-to-image diffusion models for text-to-video generation, run this command:

```bash
accelerate launch train_tuneavideo.py --config="configs/loveu-tgve-2023/DAVIS_480p/gold-fish.yaml"
```

### Automatic metrics

We employ [UMTScore]((https://github.com/llyx97/FETV)) for textual alignment evaluation and [CLIPScore](https://github.com/showlab/loveu-tgve-2023) for frame consistency evaluation.

```bibtex
@article{liu2023fetv,
  title   = {FETV: A Benchmark for Fine-Grained Evaluation of Open-Domain Text-to-Video Generation},
  author  = {Yuanxin Liu and Lei Li and Shuhuai Ren and Rundong Gao and Shicheng Li and Sishuo Chen and Xu Sun and Lu Hou},
  year    = {2023},
  journal = {arXiv preprint arXiv: 2311.01813}
}

@inproceedings{wu2023tune,
  title={Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation},
  author={Wu, Jay Zhangjie and Ge, Yixiao and Wang, Xintao and Lei, Stan Weixian and Gu, Yuchao and Shi, Yufei and Hsu, Wynne and Shan, Ying and Qie, Xiaohu and Shou, Mike Zheng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7623--7633},
  year={2023}
}

```