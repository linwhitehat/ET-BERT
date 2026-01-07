# ET-BERT: A Contextualized Datagram Representation with Pre-training Transformers for Encrypted Traffic Classification
# 微调参数实验&模型蒸馏实验

目录
=================
  * [依赖](#依赖)
  * [数据集](#数据集)
  * [使用 ET-BERT](#使用-et-bert)
  * [蒸馏 ET-BERT](#蒸馏-et-bert)

<br/>

## 依赖
* Python >= 3.6
* CUDA: 11.4
* GPU: Tesla V100S， 微调GPU：nvidia GTX5070ti
* torch >= 1.1
* six >= 1.12.0
* scapy == 2.4.4
* numpy == 1.19.2
* shutil, random, json, pickle, binascii, flowcontainer
* argparse
* packaging
* tshark
* [SplitCap](https://www.netresec.com/?page=SplitCap)
* [scikit-learn](https://scikit-learn.org/stable/)
<br/>

## 数据集
CSNET-TLS 1.3 该数据集收集于2021年3月至7月的中国科技网（CSTNET）。详见 ([CSTNET-TLS 1.3](CSTNET-TLS%201.3/readme.md))。

其它数据集可见 [paper](https://arxiv.org/abs/2202.06335) 。自制数据集请确保格式并在`data_process/`中修改路径.

<br/>

## 使用 ET-BERT
使用  [预训练模型](https://drive.google.com/file/d/1r1yE34dU2W8zSqx1FkB8gCWri4DQWVtE/view?usp=sharing) 或从此处下载:
```
wget -O pretrained_model.bin https://drive.google.com/file/d/1r1yE34dU2W8zSqx1FkB8gCWri4DQWVtE/view?usp=sharing
```

微调模型:
```
python3 fine-tuning/run_classifier.py --pretrained_model_path models/pre-trained_model.bin \
                                   --vocab_path models/encryptd_vocab.txt \
                                   --train_path datasets/cstnet-tls1.3/packet/train_dataset.tsv \
                                   --dev_path datasets/cstnet-tls1.3/packet/valid_dataset.tsv \
                                   --test_path datasets/cstnet-tls1.3/packet/test_dataset.tsv \
                                   --epochs_num 10 --batch_size 80 --embedding word_pos_seg \
                                   --encoder transformer --mask fully_visible \
                                   --seq_length 128 --learning_rate 3.3e-5
```

微调好的模型默认保存在`models/finetuned_model.bin`。测试模型：
```
python3 inference/run_classifier_infer.py --load_model_path models/finetuned_model.bin \
                                          --vocab_path models/encryptd_vocab.txt \
                                          --test_path datasets/cstnet-tls1.3/packet/nolabel_test_dataset.tsv \
                                          --prediction_path datasets/cstnet-tls1.3/packet/prediction.tsv \
                                          --labels_num 120 \
                                          --embedding word_pos_seg --encoder transformer --mask fully_visible
```
<br/>

## 蒸馏 ET-BERT
模型的蒸馏前请确保模型微调已经完成，并将教师模型保存在`models/finetuned_model.bin`。蒸馏模型：
```bash
./distillation/distill.sh
```
蒸馏好的模型默认保存在`models/distilled_student.bin`
<br/>
