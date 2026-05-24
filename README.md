# MNIST Classification with MLP and CNN from Scratch  
## 项目简介  
本项目基于 NumPy 从零实现多层感知机（MLP）与卷积神经网络（CNN），完成 MNIST 手写数字分类任务。  
项目包含：  
- MLP 与 CNN 的前向传播与反向传播实现  
- SGD、Momentum、Learning Rate Scheduling 等优化实验  
- Dropout、Weight Decay、Batch Normalization 等正则化实验  
- 模型误分类分析与权重可视化  
  
## 环境配置  
实验环境：Ubuntu 22.04.5 LTS（WSL） + Python 3.12.13。  
  
使用以下命令安装项目依赖：  
```bash
pip install -r requirements.txt
```
  
## 数据准备  
本项目未提供自动下载脚本，需要用户自行下载 MNIST 数据集。  
  
请将数据集放置到项目所需目录中，并修改代码中的路径配置。  
  
## 运行方式  
### 模型训练  
训练脚本：  
```bash
python test_train.py
```
  
请在脚本中修改对应实验配置（模型结构、学习率、Momentum、Dropout、BN 等）后运行。  
  
### 模型测试  
测试脚本：  
```bash
python test_model.py
```
  
运行前需要在源文件中修改：  
```python
# 实验目录(.pickle 文件所在目录)
RUN_DIR = r'best_models'
```
  
### 可视化脚本  
误分类可视化：  
```bash
python error_visualization.py
```
  
权重可视化：  
```bash
python weight_visualization.py
```
  
运行前需要在对应脚本中修改模型路径：  
```python
# 指向模型 pickle 文件
model_path = r'best_models/best_model.pickle'
```
  
## 模型权重  
模型权重与对应元数据下载地址：  
- [Neural Network Checkpoints for MNIST Classification](https://modelscope.cn/models/ldkevin23307130031/deep-learning-PJ1)
