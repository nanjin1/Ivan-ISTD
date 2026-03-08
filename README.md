# Ivan-ISTD
# Overview
<!-- 这是一张图片，ocr 内容为： -->
![The overall framework of our method. Stage I presents the complete Wavelet-guided Cross-domain Synthesis framework, consisting of (a) and (b). (a) illustrates the process of extracting high-value small targets from the source set (right image) and unsupervised background sample extraction (left image). (b) describes the synthesis of new samples using SSIM ranking and Poisson blending techniques. Stage II outlines the process of Real-domain Noise Invariance Learning. It involves unsupervised real noise sampling from target images, minimizing loss to enhance the model's robustness to cross-domain noise. (c) shows the detailed process of extracting noise from the target set.](https://cdn.nlark.com/yuque/0/2025/png/50405538/1760256985485-17b52b81-f596-4b96-956e-6f5df14a895c.png?x-oss-process=image%2Fformat%2Cwebp)
*Figure 1. The overall framework of our method. Stage I presents the complete Wavelet-guided Cross-domain Synthesis framework, consisting of (a) and (b). (a) illustrates the process of extracting high-value small targets from the source set (right image) and unsupervised background sample extraction (left image). (b) describes the synthesis of new samples using SSIM ranking and Poisson blending techniques. Stage II outlines the process of Real-domain Noise Invariance Learning. It involves unsupervised real noise sampling from target images, minimizing loss to enhance the model's robustness to cross-domain noise. (c) shows the detailed process of extracting noise from the target set.*
# <font style="color:rgb(31, 35, 40);">Dependencies and Installation</font>
```plain
pip install -r requirements.txt
```

# Usage
<font style="color:rgb(31, 35, 40);">Download the dataset Ivan-ISTD：Download dir[</font>[Dataset](https://pan.baidu.com/s/1t1_TwM3-Ozadu3R6eBh3rA?pwd=gi4c)<font style="color:rgb(31, 35, 40);">]</font>

<!-- 这是一张图片，ocr 内容为： -->
![(a) The diagram illustrates the process of constructing the dataset. (b) The diagram displays the composition and distribution of the dataset, where Severely Degraded Samples are created by combining Strongly Blurred Samples with Noisy Samples. (c) The diagram provides examples of backgrounds and targets across various scenes.](https://cdn.nlark.com/yuque/0/2026/png/50405538/1772944798114-99983806-b515-4669-bbe6-6db17c30b42b.png)
*Figure 2. (a) The diagram illustrates the process of constructing the dataset. (b) The diagram displays the composition and distribution of the dataset, where Severely Degraded Samples are created by combining Strongly Blurred Samples with Noisy Samples. (c) The diagram provides examples of backgrounds and targets across various scenes.*
### <font style="color:rgb(31, 35, 40);">1.Dataset</font>
+ **<font style="color:rgb(31, 35, 40);">Our project has the following structure</font>**

```plain
 ├──./datasets/
 │    ├── AugBlur-ISTD
 │    │    ├── images
 │    │    │    ├── 000001.png
 │    │    │    ├── 000002.png
 │    │    │    ├── ...
 │    │    ├── masks
 │    │    │    ├── 000001.png
 │    │    │    ├── 000002.png
 │    │    │    ├── ...
 │    │    ├── img_idx
 │    │    │    ├── train_AugBlur-ISTD.txt
 │    │    │    ├── test_AugBlur-ISTD.txt
```

### 2.Code Demo
+ **<font style="color:rgb(31, 35, 40);">EAHS</font>**

```plain
python pre_dataset/EAHS.py
```

+ **<font style="color:rgb(31, 35, 40);">BRD</font>**

```plain
python pre_dataset/BRD.py
```

+ **<font style="color:rgb(31, 35, 40);">Possion Ranker</font>**

```plain
python pre_dataset/Possion_Ranker.py
```

+ **<font style="color:rgb(31, 35, 40);">Noise Sample</font>**

```plain
python pre_dataset/Noise_Sample.py
```

### 3.Train
```plain
python Infrared_detection_model/train_ns.py
```

### 4.Test
```plain
python Infrared_detection_model/test.py
```



# <font style="color:rgb(31, 35, 40);">Results and Trained Models</font>
### <font style="color:rgb(31, 35, 40);">Visual</font>
<!-- 这是一张图片，ocr 内容为： -->
![Visualization of different models on the dataset. Blue, yellow, and red circles indicate correctly detected targets, missed detections, and false positives, respectively. The blue boxes highlight small targets for clearer observation.](https://cdn.nlark.com/yuque/0/2025/png/50405538/1743506123471-92ed2472-74c5-4f1c-b319-fc4622ef7191.png)
*Figure 3. Visualization of different models on the dataset. Blue, yellow, and red circles indicate correctly detected targets, missed detections, and false positives, respectively. The blue boxes highlight small targets for clearer observation.*


#### <font style="color:rgb(31, 35, 40);">Qualitative Results</font>
| Model | Performance |  |  |  |  |
| :---: | :---: | --- | --- | --- | --- |
|  | <font style="color:rgb(64, 64, 64);">PixAcc</font> | <font style="color:rgb(64, 64, 64);">mIoU</font> | <font style="color:rgb(64, 64, 64);">nIoU</font> | <font style="color:rgb(64, 64, 64);">Pd</font> | <font style="color:rgb(64, 64, 64);">F1</font> |
| ACM-Net | <font style="color:rgb(64, 64, 64);">19.46</font> | <font style="color:rgb(64, 64, 64);">17.62</font> | <font style="color:rgb(64, 64, 64);">19.52</font> | <font style="color:rgb(64, 64, 64);">27.34</font> | <font style="color:rgb(64, 64, 64);">18.49</font> |
| ALC-Net | <font style="color:rgb(64, 64, 64);">74.67</font> | <font style="color:rgb(64, 64, 64);">10.49</font> | <font style="color:rgb(64, 64, 64);">22.19</font> | <font style="color:rgb(64, 64, 64);">15.95</font> | <font style="color:rgb(64, 64, 64);">18.40</font> |
| DNA-Net | <font style="color:rgb(64, 64, 64);">78.41</font> | <font style="color:rgb(64, 64, 64);">71.97</font> | <font style="color:rgb(64, 64, 64);">56.46</font> | <font style="color:rgb(64, 64, 64);">56.71</font> | <font style="color:rgb(64, 64, 64);">75.05</font> |
| RDIAN | <font style="color:rgb(64, 64, 64);">65.01</font> | <font style="color:rgb(64, 64, 64);">58.03</font> | <font style="color:rgb(64, 64, 64);">46.14</font> | <font style="color:rgb(64, 64, 64);">50.25</font> | <font style="color:rgb(64, 64, 64);">61.32</font> |
| ISTDU-Net | <font style="color:rgb(64, 64, 64);">74.14</font> | <font style="color:rgb(64, 64, 64);">69.03</font> | <font style="color:rgb(64, 64, 64);">53.84</font> | <font style="color:rgb(64, 64, 64);">52.91</font> | <font style="color:rgb(64, 64, 64);">71.49</font> |
| UIU-Net | <font style="color:rgb(64, 64, 64);">58.13</font> | <font style="color:rgb(64, 64, 64);">55.96</font> | <font style="color:rgb(64, 64, 64);">41.22</font> | <font style="color:rgb(64, 64, 64);">37.47</font> | <font style="color:rgb(64, 64, 64);">57.02</font> |
| HCF-Net | <font style="color:rgb(64, 64, 64);">69.27</font> | <font style="color:rgb(64, 64, 64);">48.66</font> | <font style="color:rgb(64, 64, 64);">50.63</font> | <font style="color:rgb(64, 64, 64);">55.33</font> | <font style="color:rgb(64, 64, 64);">57.17</font> |
| SCTransNet | <font style="color:#000000;">78.37</font> | <font style="color:#000000;">74.51</font> | <font style="color:#000000;">58.75</font> | <font style="color:#000000;">60.51</font> | <font style="color:#000000;">84.47</font> |
| BGM | <font style="color:rgb(64, 64, 64);">78.43</font> | <font style="color:rgb(64, 64, 64);">72.76</font> | <font style="color:rgb(64, 64, 64);">57.93</font> | <font style="color:rgb(64, 64, 64);">62.03</font> | <font style="color:rgb(64, 64, 64);">70.33</font> |
| MSDA-Net | <font style="color:rgb(64, 64, 64);">80.24</font> | <font style="color:rgb(64, 64, 64);">75.83</font> | <font style="color:rgb(64, 64, 64);">62.29</font> | <font style="color:rgb(64, 64, 64);">66.05</font> | <font style="color:rgb(64, 64, 64);">46.83</font> |
| **Ours** | <font style="color:rgb(64, 64, 64);">81.70</font> | **<font style="color:#df2a3f;">76.01</font>** | **<font style="color:#df2a3f;">62.97</font>** | **<font style="color:#df2a3f;">68.10</font>** | **<font style="color:#df2a3f;">85.51</font>** |



# <font style="color:rgb(31, 35, 40);">Acknowledgement</font>
<font style="color:rgb(31, 35, 40);">This project is build based on</font><font style="color:rgb(31, 35, 40);"> </font>[SCTransNet](https://github.com/xdFai/SCTransNet)<font style="color:rgb(31, 35, 40);">. Thanks to Shuai Yuan.</font>

<font style="color:rgb(31, 35, 40);">The original dataset used in this study is sourced from </font>[HIT-UAV.](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset)<font style="color:rgb(31, 35, 40);"> ，with additional data collection and manual annotation.Thanks to Jiashun Suo.</font>

