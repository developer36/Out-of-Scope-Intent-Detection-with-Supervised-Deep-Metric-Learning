## Out-of-Scope-Intent-Detection-with-Supervised-Deep-Metric-Learning
### Introduction
Out-of-Scope(OOS) intents in dialogue systems is a challenging technique with practical applications. As for OOS intent detection, it not only ensures the accuracy of classifying known intents but detecting OOS intents is also crucial. Current related models are limited in learning decision boundaries or setting the threshold of confidence score, which all neglect that a well-formed intent representation is a key point. Meanwhile, text extractors trained by traditional cross- entropy loss merely focus on reducing the error rate of the class to which the sample is classified. 
In this paper, we propose an effective feature extraction method based on deep metric learning to construct the triplet network with prior knowledge. With the constructed triplet loss, mining hard samples, which refers to the far-apart intents between the same class and close intent representations among different classes, can further obtain discriminative intent representations. In addition, we also introduce adversarial training to make intent representations more robust. Experiments on three public datasets prove the effectiveness of our proposed method of learning discriminative intent representations.
This repository provides the official PyTorch implementation of the research paper [Out-of-Scope Intent Detection with Supervised Deep Metric Learning](./paper/Out-Of-Scope Intent Detection with Supervised Deep Metric Learning.pdf) (**Accepted by [IJCNN2023](https://2023.ijcnn.org/)**).

### Dependencies 

We use anaconda to create python environment:
```
conda create --name python=3.6
```
Install all required libraries:
```
pip install -r requirements.txt
```
### Run
sh run.sh

### Experiment
#### Dataset
To demonstrate the improvements of the proposed approach, we perform our experiments on three public dialogue datasets. 
* Banking: Banking is a fine-grained intent detection dataset in the banking domain.
* OOS: OOS is commonly used in text classification and open intent detection with a relative large number intent classes which is 150.
* Snips: Snips is collected by personal voice assistant which contains 7 types of user intents across different domains.
You can 
The detailed information is shown in Table.

|  Dataset   | Classes  | Training  | Validation  | Test  |
|  ----  | ----  |----  |----  |----  |
| Snips  | 7 |13,084 |700 |700 |
| Banking  | 77 |9,003 |1,000 |3080 |
| OOS  | 150 |15,000 |2,000 |6,000 |

#### Result
Macro-f1 score of unknown(open) intents and known intents with three different proportions of known classes on Banking, OOS, and Snips datasets.

| | | Banking     |  | OOS      |  |  Snips     |  |  
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| KIR | Methods | Open | Known | Open | Known | Open | Known | 
|25%| MSP      |42.17| 51.75 |62.83  | 52.81 | 0.00 | 56.71 | 
| | DOC         | 78.17 | 65.68 | 91.05 | 75.79 | 22.57 |60.53 | 
| | SEG         | 54.93 | 51.81 | 64.29 | 49.07 | 59.90 | 71.48|  
| | (K+1)-way   | 82.77 | 68.83 | 91.89 | 76.68 | 65.78 | 71.51 | 
| | ADB  | 84.24 | 70.24 | 91.96 | 77.12 | 70.17 | 73.56 | 
| | Ours     | __87.18__ | __73.62__ | __93.31__ | __79.62__ | __70.82__ |__74.62__|
|50%| MSP      |45.91| 73.14 |64.04  | 72.82 | 10.03 | 78.24 | 
| | DOC      | 72.42 | 78.25 | 87.26 | 83.79 | 49.46 |83.51 | 
| | SEG  | 43.73 | 63.66 | 59.94 | 62.34 | 25.73 | 77.48|  
| | (K+1)-way  | 74.44 | 79.01 | 88.55 | 84.83 | 70.63 | 86.59 | 
| | ADB  | 78.84 | 81.10 | 88.35 | 84.88 | 72.32 | 86.54 | 
| | Ours     | __83.55__ | __83.56__ | __90.40__ | __86.53__ | __75.46__ |__87.45__|
|75%| MSP      |48.15| 85.16 |65.81  | 84.04 | 10.12 | 86.52 | 
| | DOC      | 63.79 | 83.95 | 83.30 | 87.91 | 55.23 |90.20 | 
| | SEG  | 38.41 | 70.31 | 45.46 | 48.21 | 18.73 | 84.69|  
| | (K+1)-way  | 63.48 | 85.63 | 83.45 | 88.13 | 70.67 | 90.76 | 
| | ADB  | 66.26 | 86.09 | 85.09 | 88.99 | 71.45 | 91.16 | 
| | Ours     | __71.06__ | __87.21__ | __86.61__ | __89.65__ | __74.28__ |__91.37__|

“Open” and “Known” denote the macro f1-score over open class and known classes respectively.

If you are insterested in this work, and want to use the codes or results in this repository, please **star** this repository and **cite** by:
```
@article{Out-Of-Scope Intent Detection with Supervised Deep Metric Learning, 
      title={Out-Of-Scope Intent Detection with Supervised Deep Metric Learning}
}
```