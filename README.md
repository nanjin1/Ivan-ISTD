# Ivan-ISTD
# Overview
![](https://cdn.nlark.com/yuque/0/2025/png/50405538/1760256985485-17b52b81-f596-4b96-956e-6f5df14a895c.png?x-oss-process=image%2Fformat%2Cwebp)
# <font style="color:rgb(31, 35, 40);">Dependencies and Installation</font>
```plain
pip install -r requirement.txt
```

# Usage
<font style="color:rgb(31, 35, 40);">Download the dataset AugBlur-ISTD：Download dir[</font>[Dataset](https://pan.baidu.com/s/1t1_TwM3-Ozadu3R6eBh3rA?pwd=gi4c)<font style="color:rgb(31, 35, 40);">]</font>

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
![](https://cdn.nlark.com/yuque/0/2025/png/50405538/1743506123471-92ed2472-74c5-4f1c-b319-fc4622ef7191.png)

#### <font style="color:rgb(31, 35, 40);">Qualitative Results</font>
| <font style="color:rgb(64, 64, 64);">Model</font> | <font style="color:rgb(64, 64, 64);">Performance</font> | | | | |
| :---: | :---: | --- | --- | --- | --- |
| | <font style="color:rgb(64, 64, 64);">PixAcc</font> | <font style="color:rgb(64, 64, 64);">mIoU</font> | <font style="color:rgb(64, 64, 64);">nIoU</font> | <font style="color:rgb(64, 64, 64);">Pd</font> | <font style="color:rgb(64, 64, 64);">F1</font> |
| ACM-Net  | <font style="color:rgb(64, 64, 64);">84.59</font> | <font style="color:rgb(64, 64, 64);">46.80</font> | <font style="color:rgb(64, 64, 64);">37.84</font> | <font style="color:rgb(64, 64, 64);">38.48</font> | <font style="color:rgb(64, 64, 64);">60.25</font> |
| ALC-Net | <font style="color:rgb(64, 64, 64);">85.27</font> | <font style="color:rgb(64, 64, 64);">36.18</font> | <font style="color:rgb(64, 64, 64);">30.85</font> | <font style="color:rgb(64, 64, 64);">30.25</font> | <font style="color:rgb(64, 64, 64);">50.81</font> |
| DNA-Net | <font style="color:rgb(64, 64, 64);">79.91</font> | <font style="color:rgb(64, 64, 64);">71.30</font> | <font style="color:rgb(64, 64, 64);">57.69</font> | <font style="color:rgb(64, 64, 64);">60.00</font> | <font style="color:rgb(64, 64, 64);">75.36</font> |
| RDIAN | <font style="color:rgb(64, 64, 64);">73.95</font> | <font style="color:rgb(64, 64, 64);">62.00</font> | <font style="color:rgb(64, 64, 64);">51.20</font> | <font style="color:rgb(64, 64, 64);">59.49</font> | <font style="color:rgb(64, 64, 64);">67.45</font> |
| ISTDU-Net | <font style="color:rgb(64, 64, 64);">78.27</font> | <font style="color:rgb(64, 64, 64);">71.25</font> | <font style="color:rgb(64, 64, 64);">58.01</font> | <font style="color:rgb(64, 64, 64);">62.15</font> | <font style="color:rgb(64, 64, 64);">74.60</font> |
| UIU-Net | <font style="color:rgb(64, 64, 64);">78.28</font> | <font style="color:rgb(64, 64, 64);">69.50</font> | <font style="color:rgb(64, 64, 64);">50.43</font> | <font style="color:rgb(64, 64, 64);">59.11</font> | <font style="color:rgb(64, 64, 64);">68.43</font> |
| HCF-Net | <font style="color:rgb(64, 64, 64);">47.95</font> | <font style="color:rgb(64, 64, 64);">35.45</font> | <font style="color:rgb(64, 64, 64);">37.42</font> | <font style="color:rgb(64, 64, 64);">54.31</font> | <font style="color:rgb(64, 64, 64);">40.76</font> |
| SCTransNet | <font style="color:#000000;">78.15</font> | <font style="color:#000000;">73.93</font> | <font style="color:#000000;">60.92</font> | <font style="color:#000000;">67.21</font> | <font style="color:#000000;">83.60</font> |
| **Ours** | <font style="color:rgb(64, 64, 64);">82.54</font> | **<font style="color:#DF2A3F;">75.44</font>** | **<font style="color:#DF2A3F;">62.42</font>** | **<font style="color:#DF2A3F;">69.75</font>** | **<font style="color:#DF2A3F;">84.98</font>** |




# <font style="color:rgb(31, 35, 40);">Acknowledgement</font>
<font style="color:rgb(31, 35, 40);">This project is build based on</font><font style="color:rgb(31, 35, 40);"> </font>[SCTransNet](https://github.com/xdFai/SCTransNet)<font style="color:rgb(31, 35, 40);">. Thanks to Shuai Yuan.</font>

<font style="color:rgb(31, 35, 40);">The original dataset used in this study is sourced from </font>[HIT-UAV.](https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset)<font style="color:rgb(31, 35, 40);"> ，with additional data collection and manual annotation.Thanks to Jiashun Suo.</font>



