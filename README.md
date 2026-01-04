# DL-Final
The Task of DL-Final.

Fork本仓库，建立新分支（如`reproduction`、`model_change`、`new_model`），合并到主分支`main`请提交Pull requests.

## Progress

- [x] Baseline-1 TextCNN
- [x] Baseline-2 TextRCNN
- [ ] Model-Change
- [ ] New-Model
- [ ] Report

## Log

### @TimeAgent

- 修复 TypeError `task4\DNN-DAC\preprocess.py` make_data，以适应现使用数据集格式
- 修复 AssertionError `task4\DNN-DAC\preprocess.py` make_data，处理数据集脏数据，未知标签及空字符串""归类为'Other'
- 新增 save_test函数 `task4\DNN-DAC\train_eval.py` & `task4\DNN-DAC\run.py`，用于保存测试集结果,生成.json文件，用于提交评测
- 因无test数据集标签，原test函数无效，仅需使用save_test函数生成.json文件
- 运行`task4\DNN-DAC\run.py`后，自动train-eval-test，生成.json文件

>[!Tip]
>test输出`.json`文件位于`task4\DNN-DAC\THUCNews\log\<MODEL_NAME>\<TIME>\IMCS-DAC_test.json`。后续提交`IMCS-DAC_test.json`文件至https://tianchi.aliyun.com/competition/entrance/532044/information 进行测评。

#### Run Preprocess
进入DL-Final文件夹，所有命令基于此目录。
使用下面命令预处理数据集
```shell
python task4\DNN-DAC\preprocess.py
```
#### Run Baseline-1 TextCNN
```shell
python task4\DNN-DAC\run.py --model TextCNN
```
复现TextCNN测评结果 @Acc 0.7835 

<img src="temp/image_TextCNN.png" alt="alt text" style="zoom: 50%;" />

#### Run Baseline-2 TextRCNN
```shell
python task4\DNN-DAC\run.py --model TextRCNN 
```
复现TextRCNN测评结果 @Acc 0.7809

<img src="temp/image_TextRCNN.png" alt="alt text" style="zoom:50%;" />

### @[user-1]

#### Run Model_Change

```shell
python task4\DNN-DAC\run.py --model Model_Change
```

`Model_Change`测评结果 @Acc []

> [!Tip] 
> 测评截图存储路径及重命名：`temp/image_Model_Change.png`

<img src="temp/image_Model_Change.png" alt="alt text" style="zoom:50%;" />

### @[user-2]

#### Run New_Model

```shell
python task4\DNN-DAC\run.py --model New_Model
```

`New_Model`测评结果 @Acc []

> [!Tip] 
> 测评截图存储路径及重命名：`temp/image_New_Model.png`

<img src="temp/image_New_Model.png" alt="alt text" style="zoom:50%;" />

### @[user-3]

- 后续完成报告撰写......