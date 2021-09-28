# Image Forensics OSN

An official implementation code for paper "Robust Image Forgery Detection against Transmission over Online Social Networks"

## Table of Contents

- [Background](#background)
- [Dependency](#dependency)
- [Usage](#usage)


## Background
The increasing abuse of image editing software causes the authenticity of digital images questionable. Meanwhile, the widespread availability of online social networks (OSNs) makes them the dominant channels for transmitting forged images to report fake news, propagate rumors, etc. Unfortunately, various lossy operations, e.g., compression and resizing, adopted by OSNs impose great challenges for implementing robust image forgery detection, as shown in the below figure.

<p align='center'>  
  <img src='https://github.com/HighwayWu/ImageForensicsOSN/blob/main/imgs/demo.jpg' width='550'/>
</p>
<p align='center'>  
  <em>The detection results of DFCN (TIFS 2021) and ours by using an original forgery and the forgery transmitted through OSN. The right woman in the forgery is spliced (forged).</em>
</p>

To fight against the OSN-shared forgeries, in this work, a novel robust training scheme is proposed. Firstly, we design a baseline detector, which won top ranking in a recent certificate forgery detection competition. Then we conduct a thorough analysis of the noise introduced by OSNs, and decouple it into two parts, i.e., *predictable noise* and *unseen noise*, which are modelled separately. The former simulates the noise introduced by the disclosed (known) operations of OSNs, while the latter is designed to not only complete the previous one, but also take into account the defects of the detector itself. We finally incorporate the modelled noise into a robust training framework, significantly improving the robustness of the image forgery detector.

<p align='center'>
  <img src='https://github.com/HighwayWu/ImageForensicsOSN/blob/main/imgs/framework.jpg' width='770'/>
</p>
<p align='center'>  
  <em>The overview of our proposed training scheme and corresponding testing phase.</em>
</p>


## Dependency
- torch 1.6.0
- tensorflow 1.8.0

## Usage

For testing:
```bash
python test.py
```
Then the model will detect the images in the `data/input/` and save the results in the `data/output/` directory.
