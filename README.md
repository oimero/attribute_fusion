# 地震属性融合

## 工作流

原始训练集（30个井点）

→ 数据预处理（清除无效值较多的属性，清除井点与地震**在属性的统计指标上不符**的属性）

→ PCA降维

→ 使用 Ridge、Lasso 和 Sigmoid 拟合主成分与砂厚的关系；仅保留三模型预测最大差不超过 5m 的候选井，并以三模型均值作为虚拟井砂厚，使样本分布均衡

→ 如果样本依旧不均衡，添加随机扰动，使样本均衡

→ 根据自相关性将属性分组

→ 遍历**从多个组中选三个组**的属性组合（如果组里只有一个属性，选这个属性即可；有多个属性，则随机选择）

→ 将这三个属性作为输入，训练**组合数 x 参数网格大小**个SVR模型，取最好的五个，求平均

## 脚本用法

按顺序执行三个脚本，形成完整管线。

### 1. 井点数据预处理

```powershell
python scripts/well_data_preprocess.py \
    --input data/target/well_horizon.xlsx \
    --sheet "Sand Thickness" \
    --horizons-to-delete <要删除的层位> \
    --iqr-factor 3.0
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `data/target/well_horizon.xlsx` | 原始井点 xlsx 路径 |
| `--sheet` | `Sand Thickness` | Sheet 名称 |
| `--horizons-to-delete` | (空) | 要删除的层位列表（可多个） |
| `--wells-to-delete` | (空) | 要删除的井点列表 |
| `--iqr-factor` | `3.0` | IQR 离群值倍数 |
| `--output-name` | `well_horizon_processed.xlsx` | 输出文件名 |

输出：`scripts/output/well_data_preprocess_YYYYMMDD_HHMMSS/well_horizon_processed.xlsx` + 日志

---

### 2. PCA + Sigmoid 生成虚拟井

```powershell
python scripts/make_pesudo_sample.py \
    --seismic data/target/<层位名> \
    --wells scripts/output/<脚本1输出目录>/well_horizon_processed.xlsx \
    --surface <层位名>
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seismic` | (必填) | 地震属性 Surface 文件路径 |
| `--wells` | (必填) | 预处理后的井点 xlsx 路径（脚本 1 产出） |
| `--surface` | (必填) | 目标层位名称 |
| `--expansion-factor` | `1.5` | 工区扩展比例 |
| `--pca-variance` | `0.9` | PCA 保留方差阈值 |
| `--n-clusters` | `2` | GMM 聚类数 |
| `--sample-rows` | `40` | 采样网格行数 |
| `--sample-cols` | `40` | 采样网格列数 |
| `--max-samples-per-bin` | `30` | 每砂厚区间最大虚拟井数 |

输入依赖：脚本 1 的 `well_horizon_processed.xlsx`
输出：`scripts/output/make_pesudo_sample_YYYYMMDD_HHMMSS/<层位名>_optimized_pseudo_wells.csv` + 图件 + 中间数据

---

### 3. SVR 集成预测砂厚

```powershell
python scripts/predict_sand_thickness.py \
    --seismic data/target/<层位名> \
    --wells scripts/output/<脚本1输出目录>/well_horizon_processed.xlsx \
    --pseudo-wells scripts/output/<脚本2输出目录>/<层位名>_optimized_pseudo_wells.csv \
    --surface <层位名>
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seismic` | (必填) | 地震属性 Surface 文件路径 |
| `--wells` | (必填) | 预处理后的井点 xlsx 路径（脚本 1 产出） |
| `--pseudo-wells` | (必填) | 虚拟井 csv 路径（脚本 2 产出） |
| `--surface` | (必填) | 目标层位名称 |
| `--corr-threshold` | `0.9` | 特征相关性分组阈值 |
| `--top-models` | `5` | 集成模型数量 |

输入依赖：脚本 1 的 `well_horizon_processed.xlsx` + 脚本 2 的 `optimized_pseudo_wells.csv`
输出：`scripts/output/predict_sand_thickness_YYYYMMDD_HHMMSS/SVR_Ensemble_Prediction.txt`（可直接导入 Petrel）+ CSV + 图件

## 当前瓶颈

无论是数据扩容还是模型选择，都只能造成一些**软边界**的改变，例如原来预测薄砂的地方经过方法迭代可以预测得厚一点，但是无法影响相与相之间的**硬边界**，这种现象的本质，是训练集**样本数的不足**和**样本空间位置过于集中**，导致监督模型的预测无法泛化，使**原始地震属性之间的数据相似度**占据了最终结果的主导地位。

## 未来方向：自编码器 / 对比学习

- VAE + K-Means（[王倩楠, 王治国, 杨阳, 朱剑兵和高静怀. 《基于多特征融合自编码器的无监督地震相分类研究》. 地球物理学报 67, 期 1 (2024年1月10日): 370-78.](ref/王倩楠%20等%20-%202024%20-%20基于多特征融合自编码器的无监督地震相分类研究.pdf)）

- CNN + K-Means（[Han, Long, Xinming Wu, Zhanxuan Hu, Jintao Li, and Huijing Fang. "MAMCL: Multi-Attributes Masking Contrastive Learning for Explainable Seismic Facies Analysis." Computers & Geosciences 193 (November 2024): 105731.](ref/_han2024MAMCL.pdf)）
