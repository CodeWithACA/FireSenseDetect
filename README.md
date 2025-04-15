# 火眼金睛：去烟-融合-人体检测框架

## 简介
本项目为2025[服务外包](http://www.fwwb.org.cn/)选题[A25](http://www.fwwb.org.cn/topic/show/55e695a5-0aa6-41dc-8c95-19742cb68d05)项目，旨在提供一个基于深度学习的去烟-融合-检测框架，后续开发可根据不同的根据需求进行扩展和优化。为便于后续模型转换，本项目模型使用ONNX格式以便于拓展，在安装时可根据需求选择是否安装GPU版本onnxruntime。
[目标检测数据集和源码百度网盘](https://pan.baidu.com/s/1FM4Xi9-HIFrN2Lea2NFxEQ?pwd=xbn4 )

## 安装
本项目基于Windows系统和RK3588开发板，使用Python3.9和onnxruntime的CPU版本进行测试。
1. 克隆项目仓库：
```bash
git clone https://github.com/CodeWithACA/FireSenseDetect.git
```
2. 进入项目目录：
```bash
cd fsd
```
3. 安装依赖：
```bash
conda create -n fsd python=3.9
conda activate fsd
pip install -r requirements.txt  
```

## Windows中运行


本项目的命令行参数如下：
| 命令行参数  |是否需要| 输入 | 描述 |默认|
|---- | ---- |---|---- | ---- |
| --mode<br>-m | 是 | s/smoke/<br>fuse/f/ <br>detect /d/<br>fese_and_detect/fd | 选择运行模式 | |
| --input<br>-i | 是 | （两个）视频路径 | 若mode为fuse / f或fese_and_detect / fd 需要分别输入rgb视频路径和红外视频路径 | |
| --output<br>-o | 否 | 视频路径 | 输出视频路径 | ./result |
| --ir_ahead<br>-a|除s/d模式外均需要| float | ir视频早于rgb视频的秒数，可为负数|
| --fusefunc<br>-f| 否 | function_name | 双模态视频融合函数，可选值为densefuse, mefit |densefuse|
| --conf | 否 | float | 进行目标检测时的置信度 |0.25|
| --iou | 否 | float | 进行目标检测时的iou |0.45|
| --rknn<br>-r | 否 |  | 是否为rk3588平台 |False|

其中Mode模式提供选项和参数如下：
| 模式  |输入| 描述|
| ---- | ---- |---|
| 去烟 |  s / smoke |将输入的视频进行去烟处理|
| 融合 | fuse / f | 将不同模态视频进行融合|
| 检测 | detect / d | 将图片或视频进行直接进行检测 |
| 融合+检测 | fese_and_detect / fd | 将不同模态的视频进行融合后进行检测 |

### 去烟示例
选择模式为s/smoke 并输入视频路径
```bash
python main.py -m s -i ./demo/smoke.mp4 
```
在results文件夹下会生成去烟后的视频smoke_dehazed.mp4。此外，还可以自定义输出路径或文件名称
```bash
python main.py -m s -i ./demo/smoke.mp4 -o ./myfolder 
python main.py -m s -i ./demo/smoke.mp4 -o ./myfolder/smoke_dehazed.mp4 
```
### 融合示例
融合前必须给定ir视频早于rgb视频的秒数。对于给定的数据集，我们测量的时间如下：
| 数据集  |ir_ahead|
| ---- | ---- |
output_tr.mp4|-3.0
output_tr_smoked1.mp4|-1.0
output_tr_smoked2.mp4|-2.85
output_tr_smoked3.mp4|-4.55

demo中采用output_tr_smoked2.mp4作为示例，对应的时间为-2.85秒
```bash
python main.py -m f -i ./demo/output_rgb_smoked2.mp4  ./demo/output_tr_smoked2.mp4 -a -2.85
```
在results文件夹下会生成去烟后的视频fused_output_rgb_smoked2.mp4。此外，还可以自定义输出路径或文件名称
```bash
python main.py -m f -i ./demo/output_rgb_smoked2.mp4  ./demo/output_tr_smoked2.mp4 -a -2.85 -o myfold
python main.py -m f -i ./demo/output_rgb_smoked2.mp4  ./demo/output_tr_smoked2.mp4 -a -2.85 -o myfold/myfile.mp4
```

### 检测示例
选择模式为d/detect 并输入视频或图片路径
```bash
python main.py -m d -i ./demo/demo.jpg
python main.py -m d -i ./demo/fused_output_rgb_smoked2.mp4
```
在results文件夹下会生成检测后的视频detect_result.mp4。此外，还可以自定义输出路径或文件名称

### 融合-检测示例
选择模式为fd/fuse_and_detect 并输入视频或图片路径
```bash
python main.py -m fd -i ./demo/output_rgb_smoked2.mp4  ./demo/output_tr_smoked2.mp4 -a -2.85
```
## RK3588开发板上运行
鉴于我们提供了为RK3588开发板导出的目标检测模型，可提高在检测和融合检测模式的运行速度。当在开发板上运行且模式为d/fd是，加上-r/--rknn即可。此时需要先安装相关库。
```bash
pip install ultralytics rknn-toolkit2 rknn-toolkit-lite2
```
示例如下：
```bash
python main.py -m d -i ./demo/demo.jpg -r
python main.py -m d -i ./demo/fused_output_rgb_smoked2.mp4 -r
python main.py -m fd -i ./demo/output_rgb_smoked2.mp4  ./demo/output_tr_smoked2.mp4 -a -2.85 -r
```

## 其他
我们也提供了LatLRR算法的python版本，可使用下面的命令运行
```bash
 python ./utils/lowrank_fusion.py
 ```



## 参考
本项目参考了开源社区其他优秀项目和资料：
- [MFEIF](https://github.com/JinyuanLiu-CV/MFEIF)
- [DenseFuse](https://github.com/hli1221/densefuse-pytorch)
- [VIF-Benchmark](https://github.com/Linfeng-Tang/VIF-Benchmark)
- [ultralytics](https://github.com/ultralytics/ultralytics)
- [LatLRR](https://github.com/hli1221/imagefusion_Infrared_visible_latlrr)
- [YOLO11 目标检测 | 导出ONNX模型 | ONNX模型推理](https://blog.csdn.net/qq_41204464/article/details/142942825)
