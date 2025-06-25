import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_attribute_map(
    data_points,
    attribute_name="Predicted_Sand_Thickness",
    attribute_label="砂厚预测值(米)",
    real_wells=None,
    pseudo_wells=None,
    target_column="Sand Thickness",
    output_dir="output",
    filename_prefix="attribute_map",
    class_thresholds=[0.1, 10],  # 分类阈值：低值(<0.1)、中值(0.1-10)、高值(>10)
    figsize=(14, 14),
    dpi=300,
    cmap="viridis",
    point_size=10,
    well_size=50,
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
        alpha (float): 透明度
    """

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 检查必要的列是否存在
    required_cols = ["X", "Y", attribute_name]
    if not all(col in data_points.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data_points.columns]
        raise ValueError(f"数据中缺少必要的列: {missing}")

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
            label=f"真实井：实际值<{class_thresholds[0]}: {len(low_class)}个",
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
            label=f"真实井：实际值({class_thresholds[0]}-{class_thresholds[1]}): {len(medium_class)}个",
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
            label=f"真实井：实际值>{class_thresholds[1]}: {len(high_class)}个",
            edgecolors="white",
            linewidth=1.5,
            zorder=10,
        )

    # 如果提供了虚拟井点数据，按预测值分类并绘制
    if pseudo_wells is not None and "Mean_Pred" in pseudo_wells.columns:
        # 将平均预测值作为虚拟井的目标值
        pseudo_target = "Mean_Pred"

        # 按预测值分类
        low_pseudo = pseudo_wells[pseudo_wells[pseudo_target] < class_thresholds[0]]
        medium_pseudo = pseudo_wells[
            (pseudo_wells[pseudo_target] >= class_thresholds[0]) & (pseudo_wells[pseudo_target] <= class_thresholds[1])
        ]
        high_pseudo = pseudo_wells[pseudo_wells[pseudo_target] > class_thresholds[1]]

        # 绘制不同类别的虚拟井点
        ax.scatter(
            low_pseudo["X"],
            low_pseudo["Y"],
            color="#FF5733",  # 红橙色
            s=well_size * 0.7,  # 虚拟井点稍小一些
            marker="o",
            label=f"虚拟井：预测值<{class_thresholds[0]}: {len(low_pseudo)}个",
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
            label=f"虚拟井：预测值({class_thresholds[0]}-{class_thresholds[1]}): {len(medium_pseudo)}个",
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
            label=f"虚拟井：预测值>{class_thresholds[1]}: {len(high_pseudo)}个",
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
        loc="lower left",  # 位置改为左下角
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

    # 拆分为两张直方图：属性分布和井点分布

    # 1. 绘制属性值的直方图
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

    # 2. 如果有井点数据，单独绘制井点值的直方图
    # if (real_wells is not None and target_column in real_wells.columns) or (
    #     pseudo_wells is not None and "Mean_Pred" in pseudo_wells.columns
    # ):
    #     plt.figure(figsize=(12, 6))

    #     # 绘制真实井点数据
    #     if real_wells is not None and target_column in real_wells.columns:
    #         plt.hist(real_wells[target_column], bins=20, alpha=0.7, color="green", label="真实井")

    #     # 绘制虚拟井点数据
    #     if pseudo_wells is not None and "Mean_Pred" in pseudo_wells.columns:
    #         plt.hist(pseudo_wells["Mean_Pred"], bins=20, alpha=0.7, color="orange", label="虚拟井")

    #     # 添加标题和标签
    #     plt.xlabel("储层参数值", fontsize=12)
    #     plt.ylabel("频数", fontsize=12)
    #     plt.title("井点储层参数分布", fontsize=14)
    #     plt.grid(True, alpha=0.3)
    #     plt.legend()

    #     # 保存图像
    #     plt.savefig(
    #         os.path.join(output_dir, f"{filename_prefix}_wells_histogram.png"),
    #         dpi=dpi,
    #         bbox_inches="tight",
    #     )
    #     plt.show()
    #     plt.close()


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
                alpha=0.7,
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


def export_to_petrel_format(
    prediction_results,
    coords_columns=["X", "Y", "Z"],
    output_dir="output",
    filename_prefix="predicted",
):
    """
    将预测结果导出为Petrel可读的XYZ格式

    参数:
        prediction_results (DataFrame): 包含预测结果的数据框
        coords_columns (list): 坐标列名称
        output_dir (str): 输出目录
        filename_prefix (str): 输出文件名前缀
    """
    petrel_output_file = os.path.join(output_dir, f"{filename_prefix}_sand_thickness_petrel.txt")
    with open(petrel_output_file, "w") as f:
        # 写入标题
        f.write("# 砂厚预测结果\n")
        f.write(f"# {' '.join(coords_columns)} Predicted_Sand_Thickness\n")

        # 写入数据
        for _, row in prediction_results.iterrows():
            coords = " ".join([f"{row[col]:.6f}" for col in coords_columns])
            f.write(f"{coords} {row['Predicted_Sand_Thickness']:.6f}\n")

    print(f"预测结果已保存为Petrel可导入的XYZ格式: {petrel_output_file}")
