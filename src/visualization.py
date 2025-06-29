import os

import matplotlib.pyplot as plt
import numpy as np


def visualize_attribute_map(
    data_points,
    attribute_name="Predicted_Sand_Thickness",
    attribute_label="砂厚预测值(米)",
    real_wells=None,
    pseudo_wells=None,
    target_column="Sand Thickness",
    output_dir="output",
    filename_prefix="attribute_map",
    class_thresholds=[0.1, 10],
    figsize=(14, 14),
    dpi=300,
    cmap="viridis",
    point_size=10,
    well_size=50,
    vrange=None,  # 色彩范围元组(vmin, vmax)
):
    """
    可视化任意属性分布和井点位置

    参数:
        data_points (DataFrame): 包含坐标和属性值的数据框(必须包含X、Y和attribute_name列)
        attribute_name (str): 要可视化的属性列名
        attribute_label (str): 属性在图例和颜色条中的显示名称
        real_wells (DataFrame): 真实井点数据，需包含X、Y和target_column列
        pseudo_wells (DataFrame): 虚拟井点数据，需包含X、Y和预测列
        target_column (str): 目标列名称（如砂厚）
        output_dir (str): 输出目录
        filename_prefix (str): 输出文件名前缀
        class_thresholds (list): 分类阈值，默认[0.1, 25]表示<0.1为低值，0.1-25为中值，>25为高值
        figsize (tuple): 图像尺寸
        dpi (int): 图像分辨率
        cmap (str): 颜色图谱
        point_size (int): 数据点大小
        well_size (int): 井点标记大小
        vrange (tuple): 色彩范围元组(vmin, vmax)，None则使用数据自身范围
    """

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 检查必要的列是否存在
    required_cols = ["X", "Y", attribute_name]
    if not all(col in data_points.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data_points.columns]
        raise ValueError(f"数据中缺少必要的列: {missing}")

    # 解析vrange参数
    vmin, vmax = None, None
    if vrange is not None:
        vmin, vmax = vrange

    # 创建图像
    fig, ax = plt.subplots(figsize=figsize)

    # 绘制属性分布散点图（背景）
    scatter = ax.scatter(
        data_points["X"],
        data_points["Y"],
        c=data_points[attribute_name],
        cmap=cmap,
        s=point_size,
        marker="s",
        vmin=vmin,
        vmax=vmax,
    )

    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax, label=attribute_label)
    cbar.ax.tick_params(labelsize=10)

    # 如果提供了真实井点数据，按目标值分类并绘制
    if real_wells is not None and target_column in real_wells.columns:
        # 按目标值分类
        low_class = real_wells[real_wells[target_column] < class_thresholds[0]]
        medium_class = real_wells[
            (real_wells[target_column] >= class_thresholds[0]) & (real_wells[target_column] <= class_thresholds[1])
        ]
        high_class = real_wells[real_wells[target_column] > class_thresholds[1]]

        # 绘制不同类别的真实井点
        ax.scatter(
            low_class["X"],
            low_class["Y"],
            color="#FF5733",  # 红橙色
            s=well_size,
            marker="^",
            label=f"真实井：厚度 < {class_thresholds[0]}m: {len(low_class)}个",
            edgecolors="white",
            linewidth=1.5,
            zorder=10,
        )

        ax.scatter(
            medium_class["X"],
            medium_class["Y"],
            color="#FFFF00",  # 黄色
            s=well_size,
            marker="^",
            label=f"真实井：厚度在 {class_thresholds[0]}m-{class_thresholds[1]}m: {len(medium_class)}个",
            edgecolors="white",
            linewidth=1.5,
            zorder=10,
        )

        ax.scatter(
            high_class["X"],
            high_class["Y"],
            color="#FF00FF",  # 品红色
            s=well_size,
            marker="^",
            label=f"真实井：厚度 > {class_thresholds[1]}m: {len(high_class)}个",
            edgecolors="white",
            linewidth=1.5,
            zorder=10,
        )

    # 如果提供了虚拟井点数据，按预测值分类并绘制
    if pseudo_wells is not None:
        # 自动检测预测列名
        possible_pred_columns = ["Mean_Pred", "Predicted_Sand_Thickness", "Sand_Thickness_Pred", target_column]
        pseudo_target = None

        for col in possible_pred_columns:
            if col in pseudo_wells.columns:
                pseudo_target = col
                break

        if pseudo_target is None:
            print("警告：在虚拟井数据中未找到合适的预测列，跳过虚拟井显示")
        else:
            # 按预测值分类
            low_pseudo = pseudo_wells[pseudo_wells[pseudo_target] < class_thresholds[0]]
            medium_pseudo = pseudo_wells[
                (pseudo_wells[pseudo_target] >= class_thresholds[0])
                & (pseudo_wells[pseudo_target] <= class_thresholds[1])
            ]
            high_pseudo = pseudo_wells[pseudo_wells[pseudo_target] > class_thresholds[1]]

            # 绘制不同类别的虚拟井点
            ax.scatter(
                low_pseudo["X"],
                low_pseudo["Y"],
                color="#FF5733",  # 红橙色
                s=well_size * 0.7,  # 虚拟井点稍小一些
                marker="o",
                label=f"虚拟井：厚度 < {class_thresholds[0]}m: {len(low_pseudo)}个",
                edgecolors="white",
                linewidth=1,
                zorder=9,
                alpha=0.9,
            )

            ax.scatter(
                medium_pseudo["X"],
                medium_pseudo["Y"],
                color="#FFFF00",  # 黄色
                s=well_size * 0.7,
                marker="o",
                label=f"虚拟井：厚度在 {class_thresholds[0]}m-{class_thresholds[1]}m: {len(medium_pseudo)}个",
                edgecolors="white",
                linewidth=1,
                zorder=9,
                alpha=0.9,
            )

            ax.scatter(
                high_pseudo["X"],
                high_pseudo["Y"],
                color="#FF00FF",  # 品红色
                s=well_size * 0.7,
                marker="o",
                label=f"虚拟井：厚度 > {class_thresholds[1]}m: {len(high_pseudo)}个",
                edgecolors="white",
                linewidth=1,
                zorder=9,
                alpha=0.9,
            )

    # 添加图表标题和标签
    ax.set_title(f"{attribute_label}分布与井点位置", fontsize=16)
    ax.set_xlabel("X坐标", fontsize=14)
    ax.set_ylabel("Y坐标", fontsize=14)

    # 调整图例位置到左下角，并确保不被其他元素遮挡
    ax.legend(
        bbox_to_anchor=(-0.25, 0),  # 位置：x轴左移25%，y轴底部对齐
        loc="lower left",  # 图例内部的对齐方式
        fontsize=10,
        framealpha=0.9,  # 增加不透明度
        fancybox=True,  # 使用圆角边框
        shadow=True,  # 添加阴影效果
        ncol=1,  # 单列显示
    )

    # 添加网格线
    ax.grid(True, alpha=0.3, linestyle="--")

    # 调整布局以确保所有元素正常显示
    plt.tight_layout()

    # 保存图像
    plt.savefig(
        os.path.join(output_dir, f"{filename_prefix}_map_with_wells.png"),
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()

    # 绘制属性值的直方图
    plt.figure(figsize=(12, 6))
    plt.hist(data_points[attribute_name], bins=50, alpha=0.7, color="blue", label=f"{attribute_label}")

    # 添加标题和标签
    plt.xlabel(attribute_label, fontsize=12)
    plt.ylabel("频数", fontsize=12)
    plt.title(f"{attribute_label}统计分布", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # 保存图像
    plt.savefig(
        os.path.join(output_dir, f"{filename_prefix}_attribute_histogram.png"),
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


def visualize_feature_distribution(
    data,
    x_feature,
    y_feature,
    figsize=(10, 6),
    point_size=50,
    alpha=0.6,
    colormap="viridis",
    title=None,
    save_path=None,
):
    """
    通用的特征分布可视化函数

    Parameters:
    -----------
    data : pd.DataFrame
        包含特征数据的DataFrame
    x_feature : str
        x轴特征名
    y_feature : str
        y轴特征名（用作颜色映射）
    figsize : tuple, default=(10, 6)
        图形大小
    point_size : int, default=50
        点的大小
    alpha : float, default=0.6
        透明度
    colormap : str, default="viridis"
        颜色映射
    title : str, optional
        图表标题，如果None则自动生成
    save_path : str, optional
        保存路径

    Returns:
    --------
    matplotlib.figure.Figure : 生成的图形对象
    """
    plt.figure(figsize=figsize)

    # 创建散点图，颜色表示y特征
    scatter = plt.scatter(
        data[x_feature],
        data[y_feature],
        c=data[y_feature],
        s=point_size,
        alpha=alpha,
        cmap=colormap,
        edgecolors="black",
        linewidth=0.5,
    )

    plt.colorbar(scatter, label=f"{y_feature}")
    plt.xlabel(f"{x_feature}")
    plt.ylabel(f"{y_feature}")

    if title is None:
        title = f"特征分布: {x_feature} vs {y_feature}"
    plt.title(title)
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = f"样本数: {len(data)}\n"
    stats_text += f"{y_feature}范围: {data[y_feature].min():.2f} - {data[y_feature].max():.2f}\n"
    stats_text += f"{x_feature}范围: {data[x_feature].min():.2f} - {data[x_feature].max():.2f}"

    plt.text(
        0.02,
        0.98,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.gcf()


def visualize_gmm_clustering(
    clustering_results,
    output_dir="output",
    prefix="",
    well_data=None,
    target_column="Sand Thickness",
    class_thresholds=[0.1, 10],
    point_size=30,
    well_size=100,
):
    """
    在地理空间中可视化GMM聚类结果，使用图例而不是色彩棒

    参数:
        clustering_results (dict): perform_gmm_clustering函数的返回结果
        output_dir (str): 输出目录
        prefix (str): 文件名前缀
        well_data (DataFrame): 井点数据
        target_column (str): 井点目标列
        class_thresholds (list): 分类阈值
        point_size (int): 聚类点大小
        well_size (int): 井点大小

    返回:
        None
    """
    # 从聚类结果中提取数据
    result_df = clustering_results["result_df"]
    cluster_labels = clustering_results["cluster_labels"]
    n_clusters = len(clustering_results["cluster_counts"])

    # 文件名前缀
    file_prefix = f"{prefix}_{n_clusters}_clusters_" if prefix else f"{n_clusters}_clusters_"

    # 创建图形
    plt.figure(figsize=(10, 12))

    # 创建颜色映射，确保聚类颜色清晰区分
    unique_clusters = sorted(np.unique(cluster_labels))
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))

    # 为每个聚类单独绘制散点图
    for cluster in unique_clusters:
        mask = result_df["Cluster"] == cluster
        plt.scatter(
            result_df.loc[mask, "X"],
            result_df.loc[mask, "Y"],
            c=[colors[cluster]],  # 使用固定颜色
            label=f"聚类 {cluster}",
            s=point_size,
            alpha=0.85,
            # marker="s",
        )

    # 如果提供了井点数据，将井点添加到图中
    if well_data is not None and target_column in well_data.columns:
        # 根据目标值分类
        low_class = well_data[well_data[target_column] < class_thresholds[0]]
        medium_class = well_data[
            (well_data[target_column] >= class_thresholds[0]) & (well_data[target_column] <= class_thresholds[1])
        ]
        high_class = well_data[well_data[target_column] > class_thresholds[1]]

        # 绘制不同类别的井点
        if len(low_class) > 0:
            plt.scatter(
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
            plt.scatter(
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
            plt.scatter(
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

    # 添加标题和标签
    plt.title(f"GMM聚类分布 (聚类数={n_clusters})")
    plt.xlabel("X坐标")
    plt.ylabel("Y坐标")

    # 添加图例
    plt.legend(loc="best", framealpha=0.9, fontsize=10)

    # 添加网格
    plt.grid(True, alpha=0.3, linestyle="--")

    # 调整布局
    plt.tight_layout()

    # 保存图像
    plt.savefig(
        os.path.join(output_dir, f"{file_prefix}gmm_spatial.png"),
        dpi=300,
        bbox_inches="tight",
    )

    # 显示图像
    plt.show()

    # 绘制聚类数量分布
    plt.figure(figsize=(10, 6))
    ax = clustering_results["cluster_counts"].plot(kind="bar", color=colors)

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


def visualize_pca_clustering(
    clustering_results,
    pca_results,
    n_clusters,
    output_dir="output",
    prefix="",
    well_data=None,
    well_pca_features=None,
    target_column="Sand Thickness",
    class_thresholds=[0.1, 10],
):
    """
    在PCA空间中可视化GMM聚类结果

    参数:
        clustering_results (dict): perform_gmm_clustering函数的返回结果
        pca_results (dict): perform_pca_analysis函数的返回结果
        n_clusters (int): 聚类数量
        output_dir (str): 输出目录
        prefix (str): 文件名前缀
        well_data (DataFrame): 井点数据
        well_pca_features (ndarray): 井点在PCA空间的坐标，如果为None则不显示井点
        target_column (str): 井点目标列
        class_thresholds (list): 分类阈值

    返回:
        None
    """
    # 文件名前缀
    file_prefix = f"{prefix}_{n_clusters}_clusters_" if prefix else f"{n_clusters}_clusters_"

    # 获取PCA降维后的特征和聚类标签
    features_pca = pca_results["features_pca"]
    cluster_labels = clustering_results["cluster_labels"]

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
            os.path.join(output_dir, f"{file_prefix}gmm_pca_projection.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
    else:
        print("PCA特征维度小于2，无法在二维空间可视化")
