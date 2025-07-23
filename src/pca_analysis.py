import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, KernelPCA
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
    features, feature_stats, processing_report = preprocess_features(
        data, attribute_columns, missing_values=missing_values, verbose=False
    )

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

    # 可选：添加预处理信息到返回结果中
    if processing_report["data_quality_summary"]["total_outliers_processed"] > 0:
        print(f"\n数据预处理信息:")
        print(f"  处理的离群值总数: {processing_report['data_quality_summary']['total_outliers_processed']}")
        print(f"  特征保留率: {processing_report['data_quality_summary']['feature_reduction']}")

    print("======== PCA降维分析完成 ========")

    # 返回结果
    return {
        "pca": pca_final,  # 使用最终的PCA模型
        "scaler": scaler,
        "features_scaled": features_scaled,  # 标准化后的特征
        "n_components": n_components,
        "component_contributions": component_contributions,
        "features_clean": features,
        "features_pca": features_pca,
        "coords_clean": coords_clean,
        "explained_variance_ratio": pca_final.explained_variance_ratio_,
        "explained_variance_ratio_cumsum": np.cumsum(pca_final.explained_variance_ratio_),
        "feature_stats": feature_stats,  # 添加特征统计信息
        "processing_report": processing_report,  # 添加处理报告
    }


def perform_kpca_analysis(
    data,
    attribute_columns,
    n_components=2,
    kernel="rbf",
    gamma=None,
    output_dir="output",
    missing_values=[-999],
):
    """
    对地震属性数据进行核PCA (KPCA) 降维分析

    参数:
        data (DataFrame): 包含地震属性的数据框
        attribute_columns (list): 需要进行分析的属性列名列表
        n_components (int): 要保留的主成分数量，默认为2
        kernel (str): 使用的核函数 ('rbf', 'poly', 'sigmoid', 'cosine')
        gamma (float): RBF核的核系数，如果为None，则自动计算
        output_dir (str): 输出图表的目录
        missing_values (list): 缺失值标记

    返回:
        dict: 包含KPCA分析结果的字典
    """
    print("======== 核PCA (KPCA) 降维分析开始 ========")
    print(f"数据集大小: {data.shape}")
    print(f"核函数: {kernel}, 目标维度: {n_components}")

    # 预处理特征 (与PCA使用相同的流程)
    features, feature_stats, processing_report = preprocess_features(
        data, attribute_columns, missing_values=missing_values, verbose=False
    )

    # 标准化
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 应用KPCA
    kpca = KernelPCA(
        n_components=n_components,
        kernel=kernel,
        gamma=gamma,
        random_state=42,
        n_jobs=-1,  # 使用所有可用的CPU核心
    )
    features_kpca = kpca.fit_transform(features_scaled)
    print(f"KPCA降维后的特征形状: {features_kpca.shape}")

    # 注意：KPCA没有像线性PCA那样直观的“成分贡献度”或“解释方差比”。
    # 其主要优势在于捕捉非线性结构，而不是提供可解释的线性组合。
    print("注意: KPCA不提供类似PCA的直接载荷解释。")
    print("======== 核PCA (KPCA) 降维分析完成 ========")

    # 返回结果
    return {
        "kpca": kpca,
        "scaler": scaler,
        "features_scaled": features_scaled,
        "n_components": n_components,
        "features_kpca": features_kpca,
        "kernel": kernel,
        "gamma": kpca.gamma_,  # 返回实际使用的gamma值
    }


def visualize_pca_clustering(
    features_pca,
    cluster_labels,
    n_clusters,
    output_dir="output",
    prefix="",
    well_data=None,
    well_pca_features=None,
    target_column="Sand Thickness",
    class_thresholds=[1, 13.75],
):
    """
    在PCA空间中可视化GMM聚类结果

    参数:
        features_pca (ndarray): PCA降维后的特征，形状为(n_samples, n_components)
        cluster_labels (ndarray): 聚类标签数组
        n_clusters (int): 聚类数量
        output_dir (str): 输出目录
        well_data (DataFrame): 井点数据
        well_pca_features (ndarray): 井点在PCA空间的坐标，如果为None则不显示井点
        target_column (str): 井点目标列
        class_thresholds (list): 分类阈值

    返回:
        None
    """

    # 文件名前缀
    file_prefix = f"{n_clusters}_clusters_"

    # 仅当特征维度大于等于2时才可视化
    if features_pca.shape[1] >= 2:
        plt.figure(figsize=(12, 10))

        # 创建颜色映射，确保聚类颜色清晰区分
        unique_clusters = sorted(np.unique(cluster_labels))
        colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

        # 为每个聚类单独绘制散点图
        for cluster in unique_clusters:
            mask = cluster_labels == cluster
            plt.scatter(
                features_pca[mask, 0],
                features_pca[mask, 1],
                label=f"聚类 {cluster}",
                s=30,
                alpha=0.5,
            )

        # 如果提供了井点在PCA空间的坐标，将井点添加到图中
        if well_data is not None and well_pca_features is not None and target_column in well_data.columns:
            # 根据目标值分类
            low_indices = well_data[well_data[target_column] < class_thresholds[0]].index
            medium_indices = well_data[
                (well_data[target_column] >= class_thresholds[0]) & (well_data[target_column] <= class_thresholds[1])
            ].index
            high_indices = well_data[well_data[target_column] > class_thresholds[1]].index

            # 绘制不同类别的井点
            if len(low_indices) > 0:
                plt.scatter(
                    well_pca_features[low_indices, 0],
                    well_pca_features[low_indices, 1],
                    color="#FF5733",  # 红橙色
                    s=100,
                    marker="^",
                    edgecolors="white",
                    linewidth=1.5,
                    zorder=10,
                    label=f"井点：实际值<{class_thresholds[0]}",
                )

            if len(medium_indices) > 0:
                plt.scatter(
                    well_pca_features[medium_indices, 0],
                    well_pca_features[medium_indices, 1],
                    color="#FFFF00",  # 黄色
                    s=100,
                    marker="^",
                    edgecolors="white",
                    linewidth=1.5,
                    zorder=10,
                    label=f"井点：实际值({class_thresholds[0]}-{class_thresholds[1]})",
                )

            if len(high_indices) > 0:
                plt.scatter(
                    well_pca_features[high_indices, 0],
                    well_pca_features[high_indices, 1],
                    color="#FF00FF",  # 品红色
                    s=100,
                    marker="^",
                    edgecolors="white",
                    linewidth=1.5,
                    zorder=10,
                    label=f"井点：实际值>{class_thresholds[1]}",
                )

        plt.title(f"PCA空间中的聚类分布 (聚类数={n_clusters})")
        plt.xlabel("主成分1")
        plt.ylabel("主成分2")
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            os.path.join(output_dir, f"{file_prefix}pca_gmm_projection.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
    else:
        print("PCA特征维度小于2，无法在二维空间可视化")
