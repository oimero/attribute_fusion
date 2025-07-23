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
    class_thresholds=[1, 13.75],
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
    color_feature=None,  # 新增：自定义色标特征
    figsize=(10, 6),
    point_size=50,
    alpha=0.6,
    colormap="viridis",
    title=None,
    save_path=None,
    discrete_colors=False,  # 新增：是否使用离散颜色（适用于聚类标签）
    color_labels=None,  # 新增：离散颜色的标签（用于图例）
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
        y轴特征名
    color_feature : str, optional
        用作颜色映射的特征名，如果为None则使用y_feature
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
    discrete_colors : bool, default=False
        是否使用离散颜色（适用于聚类标签等分类变量）
    color_labels : dict, optional
        离散颜色的标签映射，格式为 {value: label}

    Returns:
    --------
    matplotlib.figure.Figure : 生成的图形对象
    """
    plt.figure(figsize=figsize)

    # 确定用于颜色映射的特征
    if color_feature is None:
        color_feature = y_feature
        color_label = f"{y_feature}"
    else:
        color_label = f"{color_feature}"

    # 检查颜色特征是否存在
    if color_feature not in data.columns:
        raise ValueError(f"颜色特征 '{color_feature}' 不存在于数据中")

    # 获取颜色数据
    color_data = data[color_feature]

    if discrete_colors:
        # 离散颜色模式（适用于聚类标签）
        unique_values = sorted(color_data.unique())
        n_colors = len(unique_values)

        # 获取颜色映射
        if isinstance(colormap, str):
            cmap = plt.cm.get_cmap(colormap)
            colors = [cmap(i / max(1, n_colors - 1)) for i in range(n_colors)]
        else:
            colors = colormap

        # 为每个类别单独绘制
        for i, value in enumerate(unique_values):
            mask = color_data == value
            if mask.any():
                # 确定标签
                if color_labels and value in color_labels:
                    label = color_labels[value]
                else:
                    label = f"{color_feature} = {value}"

                plt.scatter(
                    data.loc[mask, x_feature],
                    data.loc[mask, y_feature],
                    c=colors[i],
                    s=point_size,
                    alpha=alpha,
                    edgecolors="black",
                    linewidth=0.3,
                    label=label,
                )

        # 添加图例
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", framealpha=0.9)

    else:
        # 连续颜色模式（原始模式）
        scatter = plt.scatter(
            data[x_feature],
            data[y_feature],
            c=color_data,
            s=point_size,
            alpha=alpha,
            cmap=colormap,
            edgecolors="black",
            linewidth=0.3,
        )

        # 添加颜色条
        cbar = plt.colorbar(scatter, label=color_label)
        cbar.ax.tick_params(labelsize=10)

    # 设置坐标轴标签
    plt.xlabel(f"{x_feature}", fontsize=12)
    plt.ylabel(f"{y_feature}", fontsize=12)

    # 设置标题
    if title is None:
        if color_feature == y_feature:
            title = f"特征分布: {x_feature} vs {y_feature}"
        else:
            title = f"特征分布: {x_feature} vs {y_feature} (色标: {color_feature})"
    plt.title(title, fontsize=14)

    # 添加网格
    plt.grid(True, alpha=0.3)

    # 添加统计信息
    stats_text = f"样本数: {len(data)}\n"
    stats_text += f"{y_feature}范围: {data[y_feature].min():.2f} - {data[y_feature].max():.2f}\n"
    stats_text += f"{x_feature}范围: {data[x_feature].min():.2f} - {data[x_feature].max():.2f}"

    if color_feature != y_feature:
        if discrete_colors:
            stats_text += f"\n{color_feature}类别: {len(color_data.unique())} 个"
        else:
            stats_text += f"\n{color_feature}范围: {color_data.min():.2f} - {color_data.max():.2f}"

    # 根据是否有图例调整统计信息位置
    if discrete_colors:
        # 有图例时，将统计信息放在左上角
        text_x, text_y = 0.02, 0.98
        text_ha = "left"
    else:
        # 无图例时，将统计信息放在右上角
        text_x, text_y = 0.98, 0.98
        text_ha = "right"

    plt.text(
        text_x,
        text_y,
        stats_text,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        horizontalalignment=text_ha,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=10,
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return plt.gcf()

