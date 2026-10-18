# plant-disease-classifier

植物叶片病虫害识别的最小可运行复现：**合成叶片图像 → 特征提取 → 训练分类器 → 评估 → 桌面应用**。

技术栈：Python / scikit-learn / Pillow / Tkinter

> 说明：仓库自带**合成数据生成器**（按病斑颜色与纹理统计规律生成图像），
> 因此无需下载数据集即可跑通完整流程；换成真实叶片图片时，只需把 `dataset.py`
> 里的生成器替换为读取目录图片，其余代码不变。

## 流程

```
python -m app.dataset          # 生成合成叶片图像（3 类 × 120 张）到 data/images/
python -m app.cli train        # 提特征 → 训练随机森林 → 评估 → 保存模型 model/clf.pkl
python gui.py                  # 打开桌面应用：选择图片 → 显示类别与置信度
```

## 三个类别

| 类别 | 合成规律 |
| --- | --- |
| healthy 健康 | 均匀绿色，无斑点 |
| powdery_mildew 白粉病 | 叶片上叠加灰白色斑块（覆盖率高、边缘模糊） |
| leaf_rust 锈病 | 叶片上叠加橙褐色小点（斑点密集、颜色饱和度高） |

## 特征（12 维）

RGB 三通道均值 / 标准差、HSV 的 H/S/V 均值、绿色像素占比、白色像素占比、橙褐色像素占比、
斑点连通区域数量、斑点面积占比、边缘密度（Sobel 近似）。

## 评估

`python -m app.cli train` 会输出准确率、混淆矩阵与每类精确率 / 召回率，并写入 `model/report.md`。

## 测试

```bash
pip install -r requirements.txt
python -m pytest -q
```

测试覆盖：合成图像可生成、特征维度一致、模型可训练且准确率达标（合成数据上 >0.9）、
预测接口返回类别与置信度。
