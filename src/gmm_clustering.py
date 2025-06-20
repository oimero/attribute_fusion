import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def evaluate_gmm_clusters(features_pca, max_clusters=10, output_dir="output"):
    """
    对不同聚类数的GMM模型进行评估，绘制BIC和AIC曲线

    参数:
        features_pca (ndarray): PCA降维后的特征矩阵
        max_clusters (int): 最大聚类数量，默认为10
        output_dir (str): 输出图表的目录，默认为"output"

    返回:
        dict: 包含评估结果的字典
    """
    print("======== GMM聚类数评估开始 ========")

    # 确定最佳GMM聚类数
    n_components_range = range(1, max_clusters + 1)
    models = []
    bic_scores = []
    aic_scores = []

    for n_comp in n_components_range:
        # 训练GMM模型
        gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type="full",
            random_state=42,
            n_init=10,  # 多次初始化以获得更稳定的结果
        )
        gmm.fit(features_pca)

        # 保存模型和得分
        models.append(gmm)
        bic_scores.append(gmm.bic(features_pca))
        aic_scores.append(gmm.aic(features_pca))
        print(
            f"聚类数量 {n_comp}: BIC = {bic_scores[-1]:.2f}, AIC = {aic_scores[-1]:.2f}"
        )

    # 绘制BIC和AIC曲线
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(n_components_range, bic_scores, "o-", label="BIC")
    plt.axvline(
        np.argmin(bic_scores) + 1,
        linestyle="--",
        color="r",
        label=f"最佳聚类数 = {np.argmin(bic_scores) + 1}",
    )
    plt.xlabel("聚类数量")
    plt.ylabel("BIC分数")
    plt.title("BIC评分曲线")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(n_components_range, aic_scores, "o-", label="AIC")
    plt.axvline(
        np.argmin(aic_scores) + 1,
        linestyle="--",
        color="r",
        label=f"最佳聚类数 = {np.argmin(aic_scores) + 1}",
    )
    plt.xlabel("聚类数量")
    plt.ylabel("AIC分数")
    plt.title("AIC评分曲线")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "gmm_bic_aic.png"), dpi=300, bbox_inches="tight"
    )
    plt.show()

    # 计算BIC和AIC的变化率
    if len(bic_scores) > 1:
        bic_changes = np.diff(bic_scores) / np.array(bic_scores[:-1])
        aic_changes = np.diff(aic_scores) / np.array(aic_scores[:-1])

        plt.figure(figsize=(12, 6))
        plt.plot(range(2, max_clusters + 1), bic_changes, "o-", label="BIC变化率")
        plt.plot(range(2, max_clusters + 1), aic_changes, "s-", label="AIC变化率")
        plt.axhline(y=0, linestyle="--", color="gray")
        plt.xlabel("聚类数量")
        plt.ylabel("相对变化率")
        plt.title("BIC和AIC相对变化率")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(output_dir, "gmm_bic_aic_changes.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # 尝试找出变化率变化最大的点(可能的"肘部")
        if len(bic_changes) > 1:
            bic_elbow = np.argmax(np.abs(np.diff(bic_changes))) + 2
            aic_elbow = np.argmax(np.abs(np.diff(aic_changes))) + 2
            print(f"BIC变化率最大变化点: 聚类数 = {bic_elbow}")
            print(f"AIC变化率最大变化点: 聚类数 = {aic_elbow}")

    # 选择最佳模型（基于BIC）
    best_n_components = np.argmin(bic_scores) + 1
    best_gmm = models[best_n_components - 1]
    print(f"\n基于BIC的最佳聚类数: {best_n_components}")
    print(f"基于AIC的最佳聚类数: {np.argmin(aic_scores) + 1}")

    print("======== GMM聚类数评估完成 ========")

    # 返回结果
    return {
        "models": models,
        "bic_scores": bic_scores,
        "aic_scores": aic_scores,
        "best_n_components": best_n_components,
        "best_gmm": best_gmm,
    }


def perform_gmm_clustering(
    features_pca, coords, n_clusters, output_dir="output", random_state=42, prefix=""
):
    """
    执行指定聚类数的GMM聚类并可视化结果

    参数:
        features_pca (ndarray): PCA降维后的特征矩阵
        coords (DataFrame): 对应的坐标数据
        n_clusters (int): 聚类数量
        output_dir (str): 输出图表的目录，默认为"output"
        random_state (int): 随机种子，默认为42
        prefix (str): 输出文件名前缀，默认为""

    返回:
        dict: 包含GMM聚类结果的字典
    """
    print(f"======== 执行 {n_clusters} 聚类的GMM分析 ========")

    # 训练GMM模型
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=random_state,
        n_init=10,  # 多次初始化以获得更稳定的结果
    )
    gmm.fit(features_pca)

    # 获取聚类标签和概率
    cluster_labels = gmm.predict(features_pca)
    cluster_probs = gmm.predict_proba(features_pca)

    # 将聚类结果添加到原始数据中
    result_df = pd.DataFrame(
        {
            "X": coords["X"],
            "Y": coords["Y"],
            "Z": coords["Z"],
            "Cluster": cluster_labels,
        }
    )

    # 文件名前缀
    file_prefix = (
        f"{prefix}_{n_clusters}_clusters_" if prefix else f"{n_clusters}_clusters_"
    )

    # 可视化1: 空间分布
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        result_df["X"],
        result_df["Y"],
        c=result_df["Cluster"],
        cmap="viridis",
        s=30,
        alpha=0.8,
        linewidths=0.5,
    )
    plt.colorbar(scatter, label="聚类")
    plt.title(f"样本点的聚类分布 (聚类数={n_clusters})")
    plt.xlabel("X坐标")
    plt.ylabel("Y坐标")
    plt.savefig(
        os.path.join(output_dir, f"{file_prefix}gmm_spatial.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # 可视化2: PCA投影
    if features_pca.shape[1] >= 2:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            features_pca[:, 0],
            features_pca[:, 1],
            c=cluster_labels,
            cmap="viridis",
            s=30,
            alpha=0.7,
        )
        plt.colorbar(scatter, label="聚类")
        plt.title(f"PCA投影下的聚类分布 (聚类数={n_clusters})")
        plt.xlabel("主成分1")
        plt.ylabel("主成分2")
        plt.savefig(
            os.path.join(output_dir, f"{file_prefix}gmm_pca_projection.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    # 统计各聚类的样本数
    cluster_counts = result_df["Cluster"].value_counts().sort_index()

    plt.figure(figsize=(10, 6))
    cluster_counts.plot(kind="bar")
    plt.title(f"各聚类样本数量分布 (聚类数={n_clusters})")
    plt.xlabel("聚类")
    plt.ylabel("样本数量")
    plt.grid(axis="y")
    plt.savefig(
        os.path.join(output_dir, f"{file_prefix}gmm_cluster_counts.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    print("\n各聚类样本数量:")
    for cluster, count in cluster_counts.items():
        print(f"聚类 {cluster}: {count} 样本 ({count / len(result_df) * 100:.2f}%)")

    print(f"======== {n_clusters} 聚类的GMM分析完成 ========")

    # 返回结果
    return {
        "gmm": gmm,
        "cluster_labels": cluster_labels,
        "cluster_probs": cluster_probs,
        "result_df": result_df,
        "cluster_counts": cluster_counts,
    }


def augment_samples_by_pca_mixing(
    well_data,
    pca_model,
    scaler,
    cluster_results,
    attribute_columns,
    target_column="Thickness of facies(1: Fine sand)",
    min_samples_per_cluster=3,
    augmentation_factor=2.0,
    min_target_per_cluster=5,
    random_state=42,
    verbose=True,
):
    """
    在PCA空间中使用聚类均值混合法生成伪样本，并反投影到原始特征空间

    参数:
        well_data (DataFrame): 井点数据，包含地震属性和目标变量
        pca_model: 训练好的PCA模型
        scaler: 训练好的标准化模型
        cluster_results (dict): GMM聚类结果，包含cluster_labels和cluster_probs
        attribute_columns (list): 用于聚类和生成的属性列名
        target_column (str): 回归目标列名，默认为"Thickness of facies(1: Fine sand)"
        min_samples_per_cluster (int): 一个聚类内的最小样本数，默认为3
        augmentation_factor (float): 样本扩增倍数，默认为2.0
        min_target_per_cluster (int): 每个聚类的最小目标样本数，默认为5
        random_state (int): 随机种子，默认为42
        verbose (bool): 是否打印详细信息，默认为True

    返回:
        DataFrame: 扩增后的数据集，包含原始样本和伪样本
    """
    np.random.seed(random_state)

    # 复制原始数据
    well_data = well_data.copy()

    if verbose:
        print(f"使用 {len(attribute_columns)} 个属性列进行样本扩增")
        print(f"原始样本数: {len(well_data)}")
        print(f"目标扩增倍数: {augmentation_factor}倍")

    # 提取属性和目标
    X_orig = well_data[attribute_columns].values
    y_orig = well_data[target_column].values

    # 标准化原始数据
    X_scaled = scaler.transform(X_orig)

    # 降维到PCA空间
    X_pca = pca_model.transform(X_scaled)

    # 获取聚类标签和概率
    cluster_labels = cluster_results["cluster_labels"]

    # 将聚类标签添加到原始数据中
    well_data["Cluster"] = cluster_labels

    # 创建一个空的列表用于存储伪样本
    synthetic_samples = []

    # 统计每个聚类的样本数，并获取唯一的聚类标签
    cluster_counts = well_data["Cluster"].value_counts().to_dict()
    unique_clusters = sorted(cluster_counts.keys())  # 排序后的唯一聚类标签列表

    if verbose:
        print("\n各聚类样本数量:")
        for cluster in unique_clusters:
            print(f"聚类 {cluster}: {cluster_counts[cluster]} 样本")

    # 对每个存在的聚类进行处理
    for cluster_id in unique_clusters:
        # 获取当前聚类的样本
        cluster_mask = cluster_labels == cluster_id
        cluster_data = well_data[cluster_mask]

        # 如果聚类样本数为0，跳过（这应该不会发生，但保留为安全检查）
        if len(cluster_data) == 0:
            if verbose:
                print(f"聚类 {cluster_id} 没有样本，跳过")
            continue

        # 获取PCA空间中的数据点
        X_pca_cluster = X_pca[cluster_mask]

        # 计算当前聚类在PCA空间中的均值
        pca_cluster_mean = np.mean(X_pca_cluster, axis=0)

        # 计算目标值均值和标准差，用于生成新的目标值
        target_mean = cluster_data[target_column].mean()
        target_std = cluster_data[target_column].std()

        # 如果标准差为0或NaN（只有一个样本或所有样本相同），使用一个小的默认值
        if np.isnan(target_std) or target_std < 1e-6:
            target_std = 0.1 * target_mean if target_mean > 0 else 0.1

        # 基于扩增倍数计算目标样本数
        target_samples = max(
            min_target_per_cluster,  # 最小目标样本数
            int(len(cluster_data) * augmentation_factor),  # 基于倍数的目标样本数
        )

        # 计算需要生成的样本数
        n_samples = max(0, target_samples - len(cluster_data))

        if verbose:
            print(
                f"\n聚类 {cluster_id}: 原始样本数 = {len(cluster_data)}, 目标样本数 = {target_samples}, 需要生成 {n_samples} 个伪样本"
            )

        # 如果聚类样本数少于最小阈值，采用特殊处理
        if len(cluster_data) < min_samples_per_cluster:
            if verbose:
                print(f"  - 聚类 {cluster_id} 样本数过少，使用高权重的均值进行混合")

            # 对于样本过少的聚类，生成样本时更偏向均值
            mean_weight_range = (0.6, 0.9)  # 均值的权重范围，偏向更高权重
        else:
            # 对于样本充足的聚类，使用较为均衡的混合比例
            mean_weight_range = (0.3, 0.7)  # 均值的权重范围

        # 生成伪样本
        for i in range(n_samples):
            # 随机选择一个原始样本索引
            orig_idx = np.random.choice(np.where(cluster_mask)[0])
            orig_sample = well_data.iloc[orig_idx]

            # 获取原始样本在PCA空间中的表示
            pca_orig_sample = X_pca[orig_idx]

            # 确定均值的混合权重
            mean_weight = np.random.uniform(mean_weight_range[0], mean_weight_range[1])

            # 在PCA空间中混合生成新样本
            pca_new_sample = (
                mean_weight * pca_cluster_mean + (1 - mean_weight) * pca_orig_sample
            )

            # 反投影回原始空间
            new_sample_scaled = pca_model.inverse_transform(
                pca_new_sample.reshape(1, -1)
            )
            new_sample_orig = scaler.inverse_transform(new_sample_scaled).flatten()

            # 生成一个新的目标值，略微偏离原始样本的目标值
            # 使用有约束的正态分布，确保目标值在合理范围内
            orig_target = orig_sample[target_column]
            new_target_delta = np.random.normal(0, 0.2 * target_std)

            # 确保混合的目标值与原样本和聚类均值之间具有相似的关系
            target_weight = mean_weight  # 可以使用与特征相同的权重，或单独设置
            new_target_value = (
                target_weight * target_mean
                + (1 - target_weight) * orig_target
                + new_target_delta
            )

            # 确保目标值非负（如果是砂厚等物理量）
            new_target_value = max(0, new_target_value)

            # 创建新样本字典
            new_sample_dict = orig_sample.to_dict()
            for j, col in enumerate(attribute_columns):
                new_sample_dict[col] = new_sample_orig[j]
            new_sample_dict[target_column] = new_target_value
            new_sample_dict["Is_Synthetic"] = 1
            new_sample_dict["Cluster"] = cluster_id  # 确保聚类标签也被添加到扩充数据中

            synthetic_samples.append(new_sample_dict)

    # 创建伪样本的DataFrame
    if synthetic_samples:
        synthetic_df = pd.DataFrame(synthetic_samples)

        # 确保原始数据中有Is_Synthetic列
        well_data["Is_Synthetic"] = 0  # 标记为原始样本

        # 合并原始数据和伪样本
        augmented_data = pd.concat([well_data, synthetic_df], ignore_index=True)

        if verbose:
            print(
                f"\n扩增后的样本数: {len(augmented_data)} (原始: {len(well_data)}, 合成: {len(synthetic_df)})"
            )
            print(f"扩增比例: {len(augmented_data) / len(well_data):.2f}倍")
    else:
        well_data["Is_Synthetic"] = 0
        augmented_data = well_data
        if verbose:
            print("\n没有生成伪样本")

    return augmented_data


def encode_cluster_features(
    data, cluster_column="Cluster", drop_original=True, prefix="Cluster_"
):
    """
    对数据中的聚类标签进行One-Hot编码，并将编码后的特征添加到原始数据中

    参数:
        data (DataFrame): 包含聚类标签的数据框
        cluster_column (str): 聚类标签所在的列名，默认为"Cluster"
        drop_original (bool): 是否删除原始聚类列，默认为True
        prefix (str): One-Hot编码列的前缀，默认为"Cluster_"

    返回:
        DataFrame: 添加了One-Hot编码特征的数据框
    """
    if cluster_column not in data.columns:
        raise ValueError(f"列 '{cluster_column}' 不存在于数据中")

    # 复制原始数据，避免修改原始数据
    result_df = data.copy()

    # 获取唯一的聚类标签
    unique_clusters = sorted(result_df[cluster_column].unique())

    print(
        f"对'{cluster_column}'列进行One-Hot编码，发现{len(unique_clusters)}个唯一的聚类标签"
    )

    # 使用pandas的get_dummies函数进行One-Hot编码
    cluster_dummies = pd.get_dummies(result_df[cluster_column], prefix=prefix)

    # 合并One-Hot编码结果到原始数据
    result_df = pd.concat([result_df, cluster_dummies], axis=1)

    # 如果需要，删除原始聚类列
    if drop_original:
        result_df = result_df.drop(columns=[cluster_column])
        print(f"已删除原始'{cluster_column}'列")

    print(
        f"One-Hot编码后数据形状: {result_df.shape}，新增了{len(unique_clusters)}个特征列"
    )

    # 展示部分编码后的特征名
    encoded_cols = [col for col in result_df.columns if col.startswith(prefix)]
    print(f"编码后的特征列: {', '.join(encoded_cols)}")

    return result_df
