import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


def evaluate_gmm_clusters(features_for_clustering, max_clusters=10, output_dir="output"):
    """
    对不同聚类数的GMM模型进行评估，使用肘部法则确定最佳聚类数

    参数:
        features_for_clustering (ndarray): 用于聚类的特征数据，形状为(n_samples, n_features)
        max_clusters (int): 最大聚类数量，默认为10
        output_dir (str): 输出图表的目录，默认为"output"

    返回:
        dict: 包含评估结果的字典
    """
    print("======== GMM聚类数评估开始（带肘部法则）========")

    print(f"特征数据形状: {features_for_clustering.shape}")
    print(f"评估聚类数范围: 1 到 {max_clusters}")

    # 确定最佳GMM聚类数
    n_components_range = range(1, max_clusters + 1)
    models = []
    bic_scores = []
    aic_scores = []
    log_likelihoods = []

    for n_comp in n_components_range:
        # 训练GMM模型
        gmm = GaussianMixture(
            n_components=n_comp,
            covariance_type="full",
            random_state=42,
            n_init=10,  # 多次初始化以获得更稳定的结果
        )
        gmm.fit(features_for_clustering)

        # 保存模型和得分
        models.append(gmm)
        bic_scores.append(gmm.bic(features_for_clustering))
        aic_scores.append(gmm.aic(features_for_clustering))
        log_likelihoods.append(gmm.score(features_for_clustering))

        print(
            f"聚类数量 {n_comp}: BIC = {bic_scores[-1]:.2f}, AIC = {aic_scores[-1]:.2f}, LogLikelihood = {log_likelihoods[-1]:.2f}"
        )

    # ========== 肘部法则计算 ==========
    def find_elbow_point(scores, method_name):
        """使用肘部法则找到最佳点"""
        if len(scores) < 3:
            return 1

        # 方法1: 基于二阶差分的肘部检测
        first_diffs = np.diff(scores)
        second_diffs = np.diff(first_diffs)

        # 对于BIC/AIC（越小越好），我们寻找下降放缓的点
        # 对于对数似然（越大越好），我们寻找上升放缓的点
        if method_name in ["BIC", "AIC"]:
            # 寻找二阶差分最大的点（下降最快转为下降较慢的转折点）
            elbow_idx = np.argmax(second_diffs) + 2  # +2因为二阶差分比原数组少2个元素
        else:  # LogLikelihood
            # 寻找二阶差分最小的点（上升最快转为上升较慢的转折点）
            elbow_idx = np.argmin(second_diffs) + 2

        return min(elbow_idx, len(scores))

    # 计算不同方法的肘部点
    bic_elbow = find_elbow_point(bic_scores, "BIC")
    aic_elbow = find_elbow_point(aic_scores, "AIC")
    ll_elbow = find_elbow_point(log_likelihoods, "LogLikelihood")

    print(f"\n肘部法则检测结果:")
    print(f"BIC肘部点: 聚类数 = {bic_elbow}")
    print(f"AIC肘部点: 聚类数 = {aic_elbow}")
    print(f"LogLikelihood肘部点: 聚类数 = {ll_elbow}")

    # ========== 绘制评估曲线 ==========
    plt.figure(figsize=(18, 12))

    # 子图1: BIC曲线
    plt.subplot(2, 3, 1)
    plt.plot(n_components_range, bic_scores, "o-", label="BIC", linewidth=2)
    plt.axvline(bic_elbow, linestyle="--", color="r", label=f"肘部点 = {bic_elbow}")
    plt.axvline(np.argmin(bic_scores) + 1, linestyle=":", color="g", label=f"最小值 = {np.argmin(bic_scores) + 1}")
    plt.xlabel("聚类数量")
    plt.ylabel("BIC分数")
    plt.title("BIC评分曲线")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 子图2: AIC曲线
    plt.subplot(2, 3, 2)
    plt.plot(n_components_range, aic_scores, "o-", label="AIC", linewidth=2)
    plt.axvline(aic_elbow, linestyle="--", color="r", label=f"肘部点 = {aic_elbow}")
    plt.axvline(np.argmin(aic_scores) + 1, linestyle=":", color="g", label=f"最小值 = {np.argmin(aic_scores) + 1}")
    plt.xlabel("聚类数量")
    plt.ylabel("AIC分数")
    plt.title("AIC评分曲线")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 子图3: 对数似然曲线
    plt.subplot(2, 3, 3)
    plt.plot(n_components_range, log_likelihoods, "o-", label="Log Likelihood", linewidth=2)
    plt.axvline(ll_elbow, linestyle="--", color="r", label=f"肘部点 = {ll_elbow}")
    plt.axvline(
        np.argmax(log_likelihoods) + 1, linestyle=":", color="g", label=f"最大值 = {np.argmax(log_likelihoods) + 1}"
    )
    plt.xlabel("聚类数量")
    plt.ylabel("Log Likelihood")
    plt.title("对数似然曲线")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 子图4: BIC和AIC变化率
    if len(bic_scores) > 1:
        bic_changes = np.diff(bic_scores)
        aic_changes = np.diff(aic_scores)
        ll_changes = np.diff(log_likelihoods)

        plt.subplot(2, 3, 4)
        plt.plot(range(2, max_clusters + 1), bic_changes, "o-", label="BIC变化")
        plt.plot(range(2, max_clusters + 1), aic_changes, "s-", label="AIC变化")
        plt.axhline(y=0, linestyle="--", color="gray", alpha=0.5)
        plt.xlabel("聚类数量")
        plt.ylabel("分数变化")
        plt.title("BIC和AIC变化率")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 子图5: 对数似然变化率
        plt.subplot(2, 3, 5)
        plt.plot(range(2, max_clusters + 1), ll_changes, "^-", label="LogLikelihood变化", color="green")
        plt.axhline(y=0, linestyle="--", color="gray", alpha=0.5)
        plt.xlabel("聚类数量")
        plt.ylabel("似然变化")
        plt.title("对数似然变化率")
        plt.legend()
        plt.grid(True, alpha=0.3)

    # 子图6: 综合评估摘要
    plt.subplot(2, 3, 6)
    plt.axis("off")

    # 综合决策：优先考虑肘部法则，然后考虑最小值
    elbow_votes = [bic_elbow, aic_elbow, ll_elbow]
    elbow_consensus = max(set(elbow_votes), key=elbow_votes.count)  # 众数

    min_bic_n = np.argmin(bic_scores) + 1
    min_aic_n = np.argmin(aic_scores) + 1

    # 最终推荐：如果肘部点一致性较高，使用肘部点；否则使用BIC最小值
    elbow_agreement = elbow_votes.count(elbow_consensus) / len(elbow_votes)

    if elbow_agreement >= 0.6:  # 60%以上一致性
        recommended_n = elbow_consensus
        recommendation_reason = f"肘部法则共识({elbow_agreement:.1%}一致)"
    else:
        recommended_n = min_bic_n
        recommendation_reason = "BIC最小值(肘部法则不一致时的备选)"

    summary_text = f"""
聚类数评估摘要

肘部法则结果:
• BIC肘部点: {bic_elbow}
• AIC肘部点: {aic_elbow}
• LogLikelihood肘部点: {ll_elbow}
• 肘部共识: {elbow_consensus} (一致性: {elbow_agreement:.1%})

最优值结果:
• BIC最小值点: {min_bic_n}
• AIC最小值点: {min_aic_n}

最终推荐:
聚类数 = {recommended_n}
推荐理由: {recommendation_reason}

特征维度: {features_for_clustering.shape[1]}
数据点数: {len(features_for_clustering):,}
    """

    plt.text(
        0.05,
        0.95,
        summary_text,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
    )

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gmm_clustering_evaluation_with_elbow.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # 选择最佳模型
    best_gmm = models[recommended_n - 1]

    print(f"\n======== 最终推荐 ========")
    print(f"推荐聚类数: {recommended_n}")
    print(f"推荐理由: {recommendation_reason}")
    print(f"该聚类数的BIC: {bic_scores[recommended_n - 1]:.2f}")
    print(f"该聚类数的AIC: {aic_scores[recommended_n - 1]:.2f}")
    print("======== GMM聚类数评估完成 ========")

    # 返回结果
    return {
        "models": models,
        "bic_scores": bic_scores,
        "aic_scores": aic_scores,
        "log_likelihoods": log_likelihoods,
        "best_n_components": recommended_n,
        "best_gmm": best_gmm,
        "elbow_points": {
            "bic_elbow": bic_elbow,
            "aic_elbow": aic_elbow,
            "ll_elbow": ll_elbow,
            "consensus": elbow_consensus,
            "agreement": elbow_agreement,
        },
        "recommendation_reason": recommendation_reason,
        "n_features": features_for_clustering.shape[1],
        "n_data_points": len(features_for_clustering),
    }


def perform_gmm_clustering(
    features,  # 通用特征矩阵
    coords,
    n_clusters,
    random_state=42,
):
    """
    执行指定聚类数的GMM聚类

    参数:
        features (ndarray): 特征矩阵，可以是原始特征或PCA降维后的特征
        coords (DataFrame): 对应的坐标数据
        n_clusters (int): 聚类数量
        random_state (int): 随机种子，默认为42

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
    gmm.fit(features)

    # 获取聚类标签和概率
    cluster_labels = gmm.predict(features)
    cluster_probs = gmm.predict_proba(features)

    # 将聚类结果添加到原始数据中
    result_df = pd.DataFrame(
        {
            "X": coords["X"],
            "Y": coords["Y"],
            "Cluster": cluster_labels,
        }
    )

    # 统计各聚类的样本数
    cluster_counts = result_df["Cluster"].value_counts().sort_index()

    # 打印聚类统计信息
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


def visualize_gmm_clustering(
    clustering_results,
    output_dir="output",
    prefix="",
    well_data=None,
    target_column="Sand Thickness",
    class_thresholds=[1, 13.75],
    point_size=30,
    well_size=100,
    use_probability_colors=True,  # 新增参数：是否使用概率混合颜色
):
    """
    在地理空间中可视化GMM聚类结果，支持基于概率的颜色混合

    参数:
        clustering_results (dict): perform_gmm_clustering函数的返回结果
        output_dir (str): 输出目录
        prefix (str): 文件名前缀
        well_data (DataFrame): 井点数据
        target_column (str): 井点目标列
        class_thresholds (list): 分类阈值
        point_size (int): 聚类点大小
        well_size (int): 井点大小
        use_probability_colors (bool): 是否使用概率混合颜色，默认为True

    返回:
        None
    """
    # 从聚类结果中提取数据
    result_df = clustering_results["result_df"]
    cluster_labels = clustering_results["cluster_labels"]
    cluster_probs = clustering_results["cluster_probs"]  # 概率矩阵 (n_samples, n_clusters)
    n_clusters = len(clustering_results["cluster_counts"])

    # 文件名前缀
    file_prefix = f"{n_clusters}_clusters_{prefix}_" if prefix else f"{n_clusters}_clusters_"

    # 创建两个子图：一个用传统方式，一个用概率混合颜色
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # 创建基础颜色映射
    base_colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))[:, :3]  # 只取RGB，不要alpha通道

    # 确保颜色数量足够
    if n_clusters > len(base_colors):
        base_colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))[:, :3]

    # === 子图1: 传统的离散聚类颜色 ===
    ax1 = axes[0]

    # 为每个聚类单独绘制散点图
    unique_clusters = sorted(np.unique(cluster_labels))
    for cluster in unique_clusters:
        mask = result_df["Cluster"] == cluster
        ax1.scatter(
            result_df.loc[mask, "X"],
            result_df.loc[mask, "Y"],
            c=[base_colors[cluster]],
            label=f"聚类 {cluster}",
            s=point_size,
            alpha=0.85,
            edgecolors="black",
            linewidth=0.3,
        )

    ax1.set_title(f"传统聚类可视化 (聚类数={n_clusters})", fontsize=14)
    ax1.set_xlabel("X坐标", fontsize=12)
    ax1.set_ylabel("Y坐标", fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle="--")

    # === 子图2: 基于概率的混合颜色 ===
    ax2 = axes[1]

    if use_probability_colors and cluster_probs is not None:
        # 计算每个样本的混合颜色
        mixed_colors = np.zeros((len(cluster_probs), 3))  # RGB颜色矩阵

        for i in range(len(cluster_probs)):
            # 对每个样本，根据概率加权混合颜色
            sample_color = np.zeros(3)  # RGB
            for cluster_idx in range(n_clusters):
                prob = cluster_probs[i, cluster_idx]
                cluster_color = base_colors[cluster_idx]
                sample_color += prob * cluster_color

            # 归一化颜色值到[0,1]范围
            mixed_colors[i] = np.clip(sample_color, 0, 1)

        # 绘制混合颜色的散点图
        ax2.scatter(
            result_df["X"],
            result_df["Y"],
            c=mixed_colors,
            s=point_size,
            alpha=0.85,
            edgecolors="black",
            linewidth=0.3,
        )

        # 添加概率信息到标题
        ax2.set_title(f"概率混合颜色可视化 (聚类数={n_clusters})", fontsize=14)

        # 添加颜色参考图例（显示纯聚类颜色）
        for cluster in unique_clusters:
            ax2.scatter([], [], c=[base_colors[cluster]], s=100, label=f"聚类 {cluster} (纯色)", alpha=0.8)

    else:
        # 如果不使用概率颜色或概率数据不可用，使用传统方式
        for cluster in unique_clusters:
            mask = result_df["Cluster"] == cluster
            ax2.scatter(
                result_df.loc[mask, "X"],
                result_df.loc[mask, "Y"],
                c=[base_colors[cluster]],
                label=f"聚类 {cluster}",
                s=point_size,
                alpha=0.85,
                edgecolors="black",
                linewidth=0.3,
            )
        ax2.set_title(f"标准聚类可视化 (聚类数={n_clusters})", fontsize=14)

    ax2.set_xlabel("X坐标", fontsize=12)
    ax2.set_ylabel("Y坐标", fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle="--")

    # === 在两个子图上都添加井点数据 ===
    for ax in [ax1, ax2]:
        if well_data is not None and target_column in well_data.columns:
            # 根据目标值分类
            low_class = well_data[well_data[target_column] < class_thresholds[0]]
            medium_class = well_data[
                (well_data[target_column] >= class_thresholds[0]) & (well_data[target_column] <= class_thresholds[1])
            ]
            high_class = well_data[well_data[target_column] > class_thresholds[1]]

            # 绘制不同类别的井点
            if len(low_class) > 0:
                ax.scatter(
                    low_class["X"],
                    low_class["Y"],
                    color="#FF5733",  # 红橙色
                    s=well_size,
                    marker="^",
                    edgecolors="white",
                    linewidth=1.5,
                    zorder=10,
                    label=f"井点：实际值<{class_thresholds[0]}",
                )

            if len(medium_class) > 0:
                ax.scatter(
                    medium_class["X"],
                    medium_class["Y"],
                    color="#FFFF00",  # 黄色
                    s=well_size,
                    marker="^",
                    edgecolors="white",
                    linewidth=1.5,
                    zorder=10,
                    label=f"井点：实际值({class_thresholds[0]}-{class_thresholds[1]})",
                )

            if len(high_class) > 0:
                ax.scatter(
                    high_class["X"],
                    high_class["Y"],
                    color="#FF00FF",  # 品红色
                    s=well_size,
                    marker="^",
                    edgecolors="white",
                    linewidth=1.5,
                    zorder=10,
                    label=f"井点：实际值>{class_thresholds[1]}",
                )

        # 添加图例
        ax.legend(loc="best", framealpha=0.9, fontsize=9)

    # 调整布局
    plt.tight_layout()

    # 保存图像
    save_name = (
        f"{file_prefix}gmm_spatial_comparison.png" if use_probability_colors else f"{file_prefix}gmm_spatial.png"
    )
    plt.savefig(
        os.path.join(output_dir, save_name),
        dpi=300,
        bbox_inches="tight",
    )

    # 显示图像
    plt.show()

    # === 额外绘制：概率不确定性分析 ===
    if use_probability_colors and cluster_probs is not None:
        plt.figure(figsize=(15, 5))

        # 子图1: 最大概率分布
        plt.subplot(1, 3, 1)
        max_probs = np.max(cluster_probs, axis=1)
        scatter = plt.scatter(
            result_df["X"],
            result_df["Y"],
            c=max_probs,
            s=point_size,
            cmap="RdYlBu_r",  # 红色表示低确定性，蓝色表示高确定性
            alpha=0.8,
            edgecolors="black",
            linewidth=0.3,
        )
        plt.colorbar(scatter, label="最大聚类概率")
        plt.title("聚类确定性 (最大概率)", fontsize=12)
        plt.xlabel("X坐标")
        plt.ylabel("Y坐标")
        plt.grid(True, alpha=0.3)

        # 子图2: 熵（不确定性）
        plt.subplot(1, 3, 2)
        # 计算每个样本的概率分布熵
        entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10), axis=1)
        scatter = plt.scatter(
            result_df["X"],
            result_df["Y"],
            c=entropy,
            s=point_size,
            cmap="viridis",
            alpha=0.8,
            edgecolors="black",
            linewidth=0.3,
        )
        plt.colorbar(scatter, label="概率分布熵")
        plt.title("聚类不确定性 (熵)", fontsize=12)
        plt.xlabel("X坐标")
        plt.ylabel("Y坐标")
        plt.grid(True, alpha=0.3)

        # 子图3: 概率分布直方图
        plt.subplot(1, 3, 3)
        plt.hist(max_probs, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        plt.axvline(np.mean(max_probs), color="red", linestyle="--", label=f"平均最大概率: {np.mean(max_probs):.3f}")
        plt.xlabel("最大聚类概率")
        plt.ylabel("样本数量")
        plt.title("最大概率分布统计")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{file_prefix}gmm_probability_analysis.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # 打印概率统计信息
        print(f"\n======== 聚类概率统计 ========")
        print(f"平均最大概率: {np.mean(max_probs):.3f}")
        print(f"最大概率标准差: {np.std(max_probs):.3f}")
        print(f"平均熵: {np.mean(entropy):.3f}")
        print(
            f"高确定性样本 (最大概率>0.8): {np.sum(max_probs > 0.8)} ({np.sum(max_probs > 0.8) / len(max_probs) * 100:.1f}%)"
        )
        print(
            f"低确定性样本 (最大概率<0.6): {np.sum(max_probs < 0.6)} ({np.sum(max_probs < 0.6) / len(max_probs) * 100:.1f}%)"
        )

    # 绘制聚类数量分布（保持原有功能）
    plt.figure(figsize=(10, 6))
    ax = clustering_results["cluster_counts"].plot(kind="bar", color=base_colors[:n_clusters])

    # 在每个柱子上标注数量和百分比
    total = clustering_results["cluster_counts"].sum()
    for i, v in enumerate(clustering_results["cluster_counts"]):
        ax.text(i, v + 0.1, f"{v} ({v / total * 100:.1f}%)", ha="center", fontweight="bold")

    plt.title(f"各聚类样本数量分布 (聚类数={n_clusters})")
    plt.xlabel("聚类")
    plt.ylabel("样本数量")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"{file_prefix}gmm_cluster_counts.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()


# deprecated
def augment_samples_by_pca_mixing(
    well_data,
    pca_model,
    scaler,
    cluster_results,
    attribute_columns,
    target_column="Sand Thickness",
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
            pca_new_sample = mean_weight * pca_cluster_mean + (1 - mean_weight) * pca_orig_sample

            # 反投影回原始空间
            new_sample_scaled = pca_model.inverse_transform(pca_new_sample.reshape(1, -1))
            new_sample_orig = scaler.inverse_transform(new_sample_scaled).flatten()

            # 生成一个新的目标值，略微偏离原始样本的目标值
            # 使用有约束的正态分布，确保目标值在合理范围内
            orig_target = orig_sample[target_column]
            new_target_delta = np.random.normal(0, 0.2 * target_std)

            # 确保混合的目标值与原样本和聚类均值之间具有相似的关系
            target_weight = mean_weight  # 可以使用与特征相同的权重，或单独设置
            new_target_value = target_weight * target_mean + (1 - target_weight) * orig_target + new_target_delta

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
            print(f"\n扩增后的样本数: {len(augmented_data)} (原始: {len(well_data)}, 合成: {len(synthetic_df)})")
            print(f"扩增比例: {len(augmented_data) / len(well_data):.2f}倍")
    else:
        well_data["Is_Synthetic"] = 0
        augmented_data = well_data
        if verbose:
            print("\n没有生成伪样本")

    return augmented_data


# deprecated
def encode_cluster_features(data, cluster_column="Cluster", drop_original=True, prefix="Cluster_"):
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

    print(f"对'{cluster_column}'列进行One-Hot编码，发现{len(unique_clusters)}个唯一的聚类标签")

    # 使用pandas的get_dummies函数进行One-Hot编码
    cluster_dummies = pd.get_dummies(result_df[cluster_column], prefix=prefix)

    # 合并One-Hot编码结果到原始数据
    result_df = pd.concat([result_df, cluster_dummies], axis=1)

    # 如果需要，删除原始聚类列
    if drop_original:
        result_df = result_df.drop(columns=[cluster_column])
        print(f"已删除原始'{cluster_column}'列")

    print(f"One-Hot编码后数据形状: {result_df.shape}，新增了{len(unique_clusters)}个特征列")

    # 展示部分编码后的特征名
    encoded_cols = [col for col in result_df.columns if col.startswith(prefix)]
    print(f"编码后的特征列: {', '.join(encoded_cols)}")

    return result_df
