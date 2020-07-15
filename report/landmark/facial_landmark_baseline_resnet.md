# 面部关键点对齐任务测试报告

* 数据集：ibug-300w，68个面部关键点

## 1 基线模型——ResNet

* 任务类别：回归

* backbone模型：`resnet`（18、34、50、101、15），修改输出fc层`out_features = 2 * 68`

* loss：mse（l2）、mae（l1）、smooth_l1

* epoches：20

### 损失函数测试

* mse

$$\ell(x, y) = \frac{1}{N} \sum_{n} l_{n}, \quad l_{n} = (x_{n} - y_{n})^2$$

| backbone | best loss | test |
| --- | --- | --- |
| resnet18 | 0.0032 | ![](./img/landmark_baseline_resnet18_criterion_mse_0.png) |
| resnet34 | 0.0028 | ![](./img/landmark_baseline_resnet34_criterion_mse_0.png) |
| resnet50 | 0.0027 | ![](./img/landmark_baseline_resnet50_criterion_mse_0.png) |
| resnet101 | 0.0031 | ![](./img/landmark_baseline_resnet101_criterion_mse_0.png) |
| resnet152 | 0.0028 | ![](./img/landmark_baseline_resnet152_criterion_mse_0.png) |

* mae

$$\ell(x, y) = \frac{1}{N} \sum_{n} l_{n}, \quad l_{n} = |x_{n} - y_{n}|$$

| backbone | best loss | test |
| --- | --- | --- |
| resnet18 | 0.0377 | ![](./img/landmark_baseline_resnet18_criterion_mae_0.png) |
| resnet34 | 0.0259 | ![](./img/landmark_baseline_resnet34_criterion_mae_0.png) |
| resnet50 | 0.0384 | ![](./img/landmark_baseline_resnet50_criterion_mae_0.png) |
| resnet101 | 0.0363 | ![](./img/landmark_baseline_resnet101_criterion_mae_0.png) |
| resnet152 | 0.0362 | ![](./img/landmark_baseline_resnet152_criterion_mae_0.png) |

* smooth_l1

$$\ell(x, y) = \frac{1}{N} \sum_{n} l_{n}, \quad l_{n} =
\begin{cases}
    0.5 (x_{n} - y_{n})^2, & \text{if } |x_{n} - y_{n}| < 1 \\
    |x_{n} - y_{n}| - 0.5, & \text{otherwise }
\end{cases}$$

| backbone | best loss | test |
| --- | --- | --- |
| resnet18 | 0.0018 | ![](./img/landmark_baseline_resnet18_criterion_smooth_l1_0.png) |
| resnet34 | 0.0013 | ![](./img/landmark_baseline_resnet34_criterion_smooth_l1_0.png) |
| resnet50 | 0.0013 | ![](./img/landmark_baseline_resnet50_criterion_smooth_l1_0.png) |
| resnet101 | 0.0015 | ![](./img/landmark_baseline_resnet101_criterion_smooth_l1_0.png) |
| resnet152 | 0.0014 | ![](./img/landmark_baseline_resnet152_criterion_smooth_l1_0.png) |
