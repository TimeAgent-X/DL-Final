# DL-Final：医疗诊疗对话意图识别

本仓库是深度学习课程大作业项目，选题为阿里天池 **医疗诊疗对话意图识别挑战赛（IMCS-DAC）**。项目目标是对医患对话中的单句文本进行意图分类，例如将“医生：你好，咳嗽是连声咳吗？有痰吗？”识别为 `Request-Symptom`。

## 项目概览

- **任务类型**：中文医疗对话意图分类 / Dialogue Act Classification。
- **数据集**：`task4/IMCS-DAC` 下的 IMCS-DAC 训练集、验证集和测试集。
- **代码入口**：`task4/DNN-DAC/run.py`。
- **模型目录**：`task4/DNN-DAC/models`。
- **输出结果**：运行后生成可提交至比赛平台的 JSON 预测文件。

## 课程要求与完成情况

| 要求 | 当前实现 | 状态 |
| --- | --- | --- |
| 运行并分析至少两个 baseline | TextCNN、TextRCNN | ✅ 已完成 |
| 将 baseline 替换为其他语言模型并比较性能 | `Model_Change`：基于 `bert-base-chinese` | ✅ 已完成 |
| 基于课程知识自行设计深度学习模型 | `New_Model` | ✅ 已完成 |
| 完成实验报告 | 待补充 | ⏳ 进行中 |

## 仓库结构

```text
DL-Final/
├── README.md
├── task4/
│   ├── README.md                 # IMCS-DAC 任务说明
│   ├── IMCS-DAC/                 # 原始数据集
│   │   ├── IMCS-DAC_train.json
│   │   ├── IMCS-DAC_dev.json
│   │   └── IMCS-DAC_test.json
│   └── DNN-DAC/
│       ├── README.md             # DNN-DAC 子项目说明
│       ├── preprocess.py          # 数据预处理脚本
│       ├── run.py                 # 训练、验证、测试入口
│       ├── train_eval.py          # 训练评估与测试集预测保存逻辑
│       ├── utils.py               # 传统文本分类模型数据工具
│       ├── utils_bert.py          # BERT 类模型数据工具
│       ├── requirements.txt       # 依赖列表
│       └── models/
│           ├── TextCNN.py
│           ├── TextRCNN.py
│           ├── Model_Change.py
│           └── New_Model.py
└── temp/                          # 部分线上测评截图
```

> 说明：模型权重文件体积较大，当前仓库未上传权重文件；如需完全复现实验，请先按下方流程重新训练。

## 环境准备

建议使用独立 Python 环境。依赖文件位于 `task4/DNN-DAC/requirements.txt`。

```bash
cd DL-Final
pip install -r task4/DNN-DAC/requirements.txt
```

> 注意：不同机器的 CUDA、PyTorch 与 Transformers 版本可能影响复现结果。若运行 BERT 相关模型，请确保本地能够正常加载 `bert-base-chinese`。

## 快速开始

所有命令均在仓库根目录 `DL-Final` 下运行。

### 1. 数据预处理

```bash
python task4/DNN-DAC/preprocess.py
```

预处理后，训练所需数据会写入 `task4/DNN-DAC/THUCNews/data`。

### 2. 运行 baseline：TextCNN

```bash
python task4/DNN-DAC/run.py --model TextCNN
```

已记录线上测评结果：**Acc = 0.7835**。

### 3. 运行 baseline：TextRCNN

```bash
python task4/DNN-DAC/run.py --model TextRCNN
```

已记录线上测评结果：**Acc = 0.7809**。

### 4. 运行模型替换方案：Model_Change

```bash
python task4/DNN-DAC/run.py --model Model_Change
```

该方案使用 `bert-base-chinese`，并针对较小显存场景加入梯度检查点等优化。已记录线上测评结果：**Acc = 0.8006**。

### 5. 运行自定义模型：New_Model

```bash
python task4/DNN-DAC/run.py --model New_Model
```

已记录线上测评结果：**Acc = 0.8110**，是当前仓库记录的最佳提交结果。

## 模型与结果汇总

| 模型 | 类型 | 说明 | 线上 Acc |
| --- | --- | --- | --- |
| TextCNN | Baseline | 卷积神经网络文本分类模型 | 0.7835 |
| TextRCNN | Baseline | 融合循环结构与卷积/池化特征的文本分类模型 | 0.7809 |
| Model_Change | 模型替换 | 基于 `bert-base-chinese` 的预训练语言模型方案 | 0.8006 |
| New_Model | 自定义模型 | 课程设计的新模型方案 | 0.8110 |

部分测评截图保存在 `temp/` 目录，例如：

- `temp/image_TextCNN.png`
- `temp/image_TextRCNN.png`

## 输出与提交

`run.py` 会依次执行训练、验证和测试集预测保存。测试集预测结果会输出为 JSON 文件，典型路径如下：

```text
task4/DNN-DAC/THUCNews/log/<MODEL_NAME>/<TIME>/IMCS-DAC_test.json
```

将生成的 `IMCS-DAC_test.json` 上传至阿里天池比赛页面即可获取线上 Acc。由于测试集不包含标签，本仓库无法在本地计算测试集最终分数，需要以比赛平台结果为准。

## 已完成的主要改动

- 适配当前 IMCS-DAC 数据格式，修复 `preprocess.py` 中 `make_data` 的格式处理问题。
- 处理脏数据、未知标签与空字符串，将其统一归类为 `Other`。
- 在 `train_eval.py` 与 `run.py` 中加入测试集预测保存流程，用于生成比赛提交文件。
- 补充 `Model_Change` 与 `New_Model` 两个实验模型，并记录线上测评结果。

## 注意事项

1. 运行前请确认 `task4/IMCS-DAC` 数据集文件完整存在。
2. 若选择非 BERT 类模型，通常会使用 `utils.py` 构建数据；若选择 `Model_Change` 或 `New_Model`，会使用 `utils_bert.py`。
3. 当前仓库未包含训练后的模型权重文件，因此首次运行需要重新训练。
4. 测试集没有公开标签，`IMCS-DAC_test.json` 需要提交到比赛平台评测。
5. 若在 Windows 下运行，可以将命令中的 `/` 替换为 `\`。

## 参考资料

- 阿里天池比赛页面：<https://tianchi.aliyun.com/competition/entrance/532044/information>
- 原始 DNN-DAC baseline：<https://github.com/lemuria-wchen/imcs21-cblue/tree/main/task4/DNN-DAC>
- Chinese-Text-Classification-Pytorch：<https://gitee.com/qh123/Chinese-Text-Classification-Pytorch>
