import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data_utils import preprocess_features


def perform_pca_analysis(
    data,
    attribute_columns,
    variance_threshold=0.95,
    output_dir="output",
    missing_values=[-999],
):
    """
    对地震属性数据进行PCA降维分析

    参数:
        data (DataFrame): 包含地震属性的数据框
        attribute_columns (list): 需要进行分析的属性列名列表
        variance_threshold (float): PCA保留方差的阈值，默认为0.95
        output_dir (str): 输出图表的目录，默认为"output"
        missing_values (list): 要替换为NaN的值列表，默认为[-999]

    返回:
        dict: 包含PCA分析结果的字典
    """
    print("======== PCA降维分析开始 ========")
    print(f"数据集大小: {data.shape}")

    # 预处理特征
    features, _ = preprocess_features(data, attribute_columns, missing_values, verbose=False)

    # 提取对应的坐标
    coords_clean = data[["X", "Y"]]

    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 应用PCA
    pca = PCA()
    pca_result = pca.fit_transform(features_scaled)

    # 计算累积解释方差比
    explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)

    # 确定需要保留的主成分数量
    n_components = np.argmax(explained_variance_ratio_cumsum >= variance_threshold) + 1
    print(f"为保留至少{variance_threshold * 100}%的方差，需要保留{n_components}个主成分")

    # 绘制解释方差比
    plt.figure(figsize=(10, 6))
    plt.bar(
        range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
        alpha=0.8,
        label="单个方差贡献",
    )
    plt.step(
        range(1, len(pca.explained_variance_ratio_) + 1),
        explained_variance_ratio_cumsum,
        where="mid",
        label="累积方差贡献",
    )
    plt.axhline(
        y=variance_threshold,
        linestyle="--",
        color="r",
        label=f"{variance_threshold * 100}%方差阈值",
    )
    plt.axvline(
        x=n_components,
        linestyle="--",
        color="g",
        label=f"选择的主成分数({n_components})",
    )
    plt.title("PCA解释方差比")
    plt.xlabel("主成分数量")
    plt.ylabel("解释方差比")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(output_dir, "pca_explained_variance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # 使用选择的主成分数重新应用PCA
    pca_final = PCA(n_components=n_components)
    features_pca = pca_final.fit_transform(features_scaled)
    print(f"降维后的特征形状: {features_pca.shape}")

    # 打印每个主成分对应的原始特征贡献
    component_contributions = pd.DataFrame(
        pca_final.components_,
        columns=features.columns,
        index=[f"PC{i + 1}" for i in range(n_components)],
    )
    print("\n主成分与原始特征的关系:")
    print(component_contributions)

    print("======== PCA降维分析完成 ========")

    # 返回结果 - 添加缺失的键
    return {
        "pca": pca_final,  # 使用最终的PCA模型
        "scaler": scaler,
        "features_scaled": features_scaled,  # 添加标准化后的特征
        "n_components": n_components,
        "component_contributions": component_contributions,
        "features_clean": features,
        "features_pca": features_pca,
        "coords_clean": coords_clean,
        "explained_variance_ratio": pca_final.explained_variance_ratio_,  # 添加解释方差比
        "explained_variance_ratio_cumsum": np.cumsum(pca_final.explained_variance_ratio_),  # 添加累积解释方差比
    }
