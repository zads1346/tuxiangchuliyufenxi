# 基于 U-Net 的猫狗前景分割

本项目基于公开的 Oxford-IIIT Pet 数据集，使用 U-Net 完成猫狗前景二分类分割，并比较训练数据增强、水平翻转测试时增强（TTA）及连通域过滤。仓库包含实验代码、固定数据划分、数据审计记录和已完成实验的数值结果。

## 本次实验结果

训练使用相同的数据划分和超参数，两组各训练 5 轮，使用验证集 Dice 选择权重。表中数字来自实际运行记录，测试集为排除 7 张异常全背景掩码后的官方测试有效子集，含 3,662 张图片。

| 训练配置 | 测试 Dice | 测试 IoU |
| --- | ---: | ---: |
| 基线 | 0.9108 | 0.8545 |
| 训练时水平翻转与亮度增强 | 0.9071 | 0.8491 |

以下推理对照使用同一个基线权重。先用验证集选择连通域面积阈值，固定后计算测试集指标：

| 推理方式 | 测试 Dice | 测试 IoU |
| --- | ---: | ---: |
| 原始预测 | 0.910833 | 0.854537 |
| 水平翻转 TTA | 0.915008 | 0.861187 |
| TTA + 连通域过滤 | 0.915001 | 0.861304 |

训练增强在这次实验中未提高测试分数；TTA 有小幅提升，过滤效果变化很小。训练仅 5 轮且使用一次随机种子，不代表充分收敛或统计显著。推理改进是在已查看该测试集结果后开展，属于探索性复用，不是全新独立确认。

## 数据与许可

数据集来源：[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)。原论文：Parkhi O M, Vedaldi A, Zisserman A, Jawahar C V. *Cats and Dogs*. CVPR 2012, 3498–3505. [DOI: 10.1109/CVPR.2012.6248092](https://doi.org/10.1109/CVPR.2012.6248092)。

数据集官网列出 CC BY-SA 4.0，并说明图像版权属于原权利人；下载包附带的 README 则写明研究用途并要求遵守原网站条款。两处说明存在差异，使用和再分发前请核查适用条款。仓库不重新分发数据集照片、原始压缩包、含照片的报告或视频。模型权重也未包含；运行推理对照需要先训练基线模型。

请从官网获取 `images.tar.gz` 和 `annotations.tar.gz`，解压到 `data/oxford-iiit-pet/`。保留数据集原始结构：

```text
data/
  splits.json
  splits_valid.json
  oxford-iiit-pet/
    images/                         # 原图 .jpg
    annotations/
      trainval.txt
      test.txt
      trimaps/                      # 三值标注 .png
```

示例（在仓库根目录运行，Windows PowerShell）：

```powershell
New-Item -ItemType Directory -Force data/oxford-iiit-pet | Out-Null
curl.exe -L --fail -o data/images.tar.gz https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
curl.exe -L --fail -o data/annotations.tar.gz https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xzf data/images.tar.gz -C data/oxford-iiit-pet
tar -xzf data/annotations.tar.gz -C data/oxford-iiit-pet
```

训练和测试沿用官方名单，在训练部分按品种抽出验证集；`data/splits_valid.json` 进一步排除 14 张图中有宠物而掩码全为背景的图片，得到训练 3,307、验证 366、测试 3,662 张。标签 1 为前景、2 为背景、3 为未分类边界；后者不计入损失与指标。

数据审计发现 23 张测试图与训练或验证图高度相似，排除它们后另做敏感性汇总。哈希检索不能发现所有近重复或证明动物个体独立。细节见 `data/audit.json`、`data/cross_split_dhash_method.md` 和 `data/cross_split_dhash_review.json`。

## 环境与运行

实验环境：Python 3.11、PyTorch 2.5.1、NumPy 2.4.4、Pillow 12.1.1、Matplotlib 3.10.8、OpenCV 4.13，GPU 为 RTX 4060 Laptop。先根据自己的硬件安装合适的 [PyTorch](https://pytorch.org/get-started/locally/)，再安装其余依赖：

```bash
python -m pip install -r requirements.txt
python experiment.py --help
```

从仓库根目录运行。以下命令会重新训练并覆盖 `runs/` 下同名结果；首次训练会自动创建目录：

```bash
python experiment.py --variant baseline --epochs 5 --batch 4 --size 256 --seed 42
python experiment.py --variant augmented --epochs 5 --batch 4 --size 256 --seed 42
python compare.py
python sensitivity.py
python inference_improvements.py
```

脚本自动选择 CUDA（可用时）或 CPU。CPU 运行会更慢。`inference_improvements.py` 使用刚训练的 `runs/baseline/best.pt`，并与本地本次训练生成的指标做一致性提示。不同硬件或库版本、随机性可能带来不同结果；不能保证重训后逐位复现原始数字。

`data/audit_pet.py` 可以在下载原压缩包后重新审计，但会覆盖 `data/audit.json` 和划分清单，请先备份。`data/audit_cross_split_duplicates.py` 可重新计算跨集合候选；人工判断记录已随仓库保存。

## 仓库结构

- `experiment.py`：数据处理、U-Net 训练、验证、测试和逐图预测。
- `compare.py`、`sensitivity.py`：训练对照与近重复敏感性分析。
- `inference_improvements.py`：验证集选阈值及 TTA、连通域消融。
- `data/`：固定划分、审计记录及可复现脚本；不含数据集原图。
- `results/`：原实验的数值指标与逐图结果；不是可直接加载的模型权重。
- `upstream/unet/`：保留的上游 U-Net 网络实现。

上游代码来自 [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet)，具体版本和修改说明见 [SOURCE.md](SOURCE.md)。上游 GPL-3.0 许可证随仓库保留，见 [LICENSE](LICENSE)。
