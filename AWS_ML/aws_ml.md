# Exam Logistics

- 170 min
- 65 questions
- Contents:
  - Data engineering - 20%
  - Exploratory data analysis - 24%
  - Modellling - 36%
  - ML implementation & operation  20%
- Type:
  - **Multiple choice**
  - **Multiple response**

建议：

- 先读题，尝试在看选项前答题。

- 找关键词（qualifier & key phrase），并根据此去掉错误选项。

- 实在不会，先跳过。

  


## Part I: Data engineering - 20%
### 1，Create data repositories for ML

数据形式[structured, unstruced] -> a centralized repository -> Data Lake

AWS Lake Formation 

Amason S3 storage option for ds processing on AWS


### 2，Identify and implement a data-ingestion solution
### 3，Identify and implement a data-transformation solution

  

## Part II: Exploratory data analysis - 24%

## Part III: Modellling - 36%

## Part IV: ML implementation & operation  20%





# ML for Business Leaders

## When & How?

### 1,when?

when is machine learning a proporate tool to solve my problem?

what can? what tools?

<img src="awsml_pic/what_ml_can.png" alt="what_ml_can" style="zoom:50%;" />

When not to use?

no data

no groundtruth labels

quick launch

no tolerance for error

### 2, Six questions to ask?

1, what are the made assumption?

2, what is the learning target / hypothesis? (hypothesis testing for large datasets is basic promise for ML)

3, what type of ML problem is it?

4, why did you choose this algorithm? (simpler baseline?)

5, how will you evaluate the model performance?

6, how confident are you that u can generalize the results?		

### 3, How?

how to identify ML opportunities?

**Amazon ML applications:**

recommendations

robotics optimizations

forecasting

search optimizations

delivery routes

Alexa

### 4, Define and scope a ML problem?

ML is the subfield of AI, prevalence of large data sets and massive computational resources has made the domaniance the field of AI. 

![ml_map](awsml_pic/ml_map.png)

#### i,define problem

<img src="awsml_pic/define_probelm.png" alt="define_probelm" style="zoom:50%;" />

#### ii, input gathering

<img src="awsml_pic/input.png" alt="input" style="zoom:50%;" />

#### iii, output

<img src="awsml_pic/output.png" alt="output" style="zoom:50%;" />

### 5, ML Process

![MLprocess](awsml_pic/MLprocess.png)

#### Feature Engineering domain specific 

<img src="awsml_pic/domain_specific.png" alt="domain_specific" style="zoom:50%;" />

#### Parameter Tuning

- loss function [和ground truth的差别]
- regularisation [increase the generalization to better fit the data]
- learning parameters (decay rate 控制model学习的快慢)

<img src="awsml_pic/parameter_tunning.png" alt="parameter_tunning" style="zoom:50%;" />

### Evaluation 

##### 1, Overfitting vs underfitting（generalize more toward unseen data）

- use validation error 

- using training error -> overfitting, lack of feature/information -> undercutting

##### 2, Bias-variance tradeoff [supervised]

<img src="awsml_pic/bias-variance_tradeoff.png" alt="bias-variance_tradeoff" style="zoom:50%;" />

##### 3, evaluation matrix

| 模型       | evaluation                                      |                      |
| ---------- | ----------------------------------------------- | -------------------- |
| Regression | <img src="awsml_pic/regression_eva.png"  /> | -RMSE，MAPE 越大越好<br> -R^2 越大越好 |
| Classification | - confusion matrix<img src="awsml_pic/confusion_matrix.png" alt="confusion_matrix" style="zoom:50%;" /> <br/> -precision recall <img src="awsml_pic/presion_recall.png" alt="presion_recall" style="zoom:50%;" /> | - precision: how correct we are on ones we predictect would be positive <br/> - recall: fraction of negatives that we wrongly predicted<br>i.e. search engine; precision, quality and how relevant it is; completeness and fraction of relevance |
| Binary classification 例子 | - type I: alpha ~ 5%<br/>- type II: beta 1- power<br/>- power ~ 80% [依情况订]<br/><img src="awsml_pic/type12error.png" alt="type12error" style="zoom:50%;" /><img src="awsml_pic/binary.png" alt="binary"  /> <br> - specificity = TN/TN + FP <br> - FPR = 1- specificity = FP/ TN + FP<br>  | - precision：在我们判断是disease中有多少人是真的病了<br>- recall: 在有disease的样本量中，有多少我们可以正确的判断出来。<br>- accuracy: 正确判断的[overall]。<br> |
| ROC <br>AUC | <img src="awsml_pic/roc.png" alt="a" style="zoom:30%;" /><img src="awsml_pic/a.png" alt="a" style="zoom:30%;" /> | ROC: <br/>1, 选择不同的threshold，TPR 和 FPR 对应关系。<br/>2, FPR越小，TPR越大。全局最优解，能接受的FPR左边能接受的点。<br/> AUC: auc 面积越大，模型越好 |

 

### Key issues in ML

[AWS: The elements of Data Science](https://aws.amazon.com/training/learning-paths/machine-learning/exam-preparation/)

#### Data quality

consistency of the data (和问题一致么)

accuracy of the data

noisy data (fluctuation in the input and output)

missing data (那些模型对missing敏感？)

outliers 

bias

Variance

#### Model quality

overfitting and underfitting

![overandunder](awsml_pic/overandunder.png)

#### Computation speed and scalability

use sagemaker and EC2 

- increase speed

- solve prediction time complexity

- solve space complexity



## Supervised learning:

### Linear methods:

![linear](awsml_pic/linear.png)

#### Linear regression (univariate)

<img src="awsml_pic/lr.png" alt="lr" style="zoom:85%;" />

#### Multivariate LR

Multicollinearity

<img src="awsml_pic/mlr.png" alt="mlr" style="zoom:55%;" />

### Logistic regression

<img src="awsml_pic/logistic.png" alt="logistic" style="zoom:67%;" />

#### sigmoid curve

is a good representation of probability which is widely used in logistic regression to fit a model. x [-inf, inf]

<img src="awsml_pic/sigmoid.png" alt="sigmoid" style="zoom:50%;" />

#### how to fit logistic regression model

<img src="awsml_pic/fitsigmoid.png" alt="fitsigmoid" style="zoom:60%;" />

