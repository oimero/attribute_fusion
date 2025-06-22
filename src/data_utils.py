import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def identify_attributes(file_path):
    """
    从Petrel格式的地震属性文件中识别属性

    参数:
        file_path (str): Petrel文件的路径

    返回:
        tuple: (属性列表, END ATTRIBUTES所在行号)
    """
    print(f"正在识别文件属性: {file_path}")

    # 读取文件内容
    with open(file_path, "r") as file:
        lines = file.readlines()

    attr_start = -1
    attr_end = -1
    attributes = []

    # 查找属性部分的起始和结束位置
    for i, line in enumerate(lines):
        if line.strip() == "ATTRIBUTES":
            attr_start = i
        elif line.strip() == "END ATTRIBUTES":
            attr_end = i
            break

    print(f"识别到 END ATTRIBUTES 位于第 {attr_end} 行")

    # 提取属性名称
    if attr_start != -1 and attr_end != -1:
        # 跳过ATTRIBUTES行
        for i in range(attr_start + 1, attr_end):
            line = lines[i].strip()
            parts = line.split(",")
            if len(parts) >= 4:  # 确保有足够的部分
                attr_name = parts[-1].strip()  # 取最后一部分作为属性名
                attributes.append(attr_name)

    print(f"识别到 {len(attributes)} 个属性:")
    for attr in attributes:
        print(f"  - {attr}")

    return attributes, attr_end


def parse_petrel_file(file_path):
    """解析Petrel格式的地震属性文件"""
    print(f"正在解析文件: {file_path}")

    # 步骤1: 识别属性和确定跳过的行数
    attributes, skip_rows = identify_attributes(file_path)

    # 读取文件头以确定列的含义
    with open(file_path, "r") as file:
        lines = file.readlines()

    # 分析文件头确定列结构
    column_meanings = []
    for line in lines:
        if line.startswith("# Field"):
            parts = line.split(":")
            if len(parts) >= 2:
                field_name = parts[1].strip()
                column_meanings.append(field_name)

    print(f"识别到的列含义: {column_meanings}")

    # 步骤2: 构建列名
    # 前几列使用文件头中定义的名称
    column_names = column_meanings.copy() if column_meanings else ["X", "Y", "Z"]

    # 添加属性列名
    column_names.extend(attributes)

    # 确认列数
    with open(file_path, "r") as file:
        for i, line in enumerate(file):
            if i > skip_rows:  # 第一个数据行
                data_columns = line.strip().split()
                num_columns = len(data_columns)
                break

    print(f"数据行有 {num_columns} 列，列名列表有 {len(column_names)} 个")

    # 处理列数不匹配的情况
    if num_columns != len(column_names):
        print(f"警告: 列数({num_columns})与列名数({len(column_names)})不匹配!")
        # 如果列名太多，截断
        if len(column_names) > num_columns:
            column_names = column_names[:num_columns]
            print(f"截断列名列表至 {num_columns} 个")
        # 如果列名太少，添加占位符
        else:
            for i in range(len(column_names), num_columns):
                column_names.append(f"Column_{i + 1}")
            print(f"添加占位符列名至 {num_columns} 个")

    # 读取数据
    try:
        df = pd.read_csv(
            file_path,
            delim_whitespace=True,
            skiprows=skip_rows + 1,  # 跳过END ATTRIBUTES行及之前的所有行
            names=column_names,
            dtype=float,
            engine="python",
        )
        print(f"成功读取数据，共 {len(df)} 行")

        # 打印数据的基本统计信息以验证
        print("\n数据预览:")
        print(df.head())
        print("\n各列统计信息:")
        print(df.describe().T[["min", "max", "mean", "std"]])

        return df
    except Exception as e:
        print(f"读取数据时出错: {str(e)}")
        return None


def filter_anomalous_attributes(
    seismic_data,
    well_data,
    common_attributes,
    ratio_threshold=5.0,
    range_ratio_threshold=10.0,
    std_ratio_threshold=10.0,
    exclude_negative_ratio=True,
    output_dir=None,
    verbose=True,
):
    """
    评估共同属性的数据质量并筛选出统计特征异常的属性

    参数:
        seismic_data (DataFrame): 地震数据
        well_data (DataFrame): 井点数据
        common_attributes (list): 共同属性列表
        ratio_threshold (float): 均值比值阈值，超过此值被视为异常
        range_ratio_threshold (float): 数值范围比值阈值
        std_ratio_threshold (float): 标准差比值阈值
        exclude_negative_ratio (bool): 是否排除均值比值为负的属性
        output_dir (str): 输出图表的目录，如不需要可设为None
        verbose (bool): 是否打印详细信息

    返回:
        tuple: (筛选后的属性列表, 异常属性信息DataFrame, 所有属性统计比较DataFrame)
    """
    if verbose:
        print("======== 井点数据与地震数据的属性统计比较 ========")

    # 获取统计信息
    def get_stats_summary(data, attributes):
        stats = {}
        for attr in attributes:
            if attr in data.columns:
                stats[attr] = {
                    "count": data[attr].count(),
                    "mean": data[attr].mean(),
                    "std": data[attr].std(),
                    "min": data[attr].min(),
                    "25%": data[attr].quantile(0.25),
                    "median": data[attr].median(),
                    "75%": data[attr].quantile(0.75),
                    "max": data[attr].max(),
                    "skew": data[attr].skew(),
                }
        return stats

    # 获取井点数据和地震数据的统计信息
    well_stats = get_stats_summary(well_data, common_attributes)
    seismic_stats = get_stats_summary(seismic_data, common_attributes)

    # 创建比较表格
    comparison_rows = []
    for attr in common_attributes:
        if attr in well_stats and attr in seismic_stats:
            well_mean = well_stats[attr]["mean"]
            seismic_mean = seismic_stats[attr]["mean"]
            well_std = well_stats[attr]["std"]
            seismic_std = seismic_stats[attr]["std"]
            well_range = well_stats[attr]["max"] - well_stats[attr]["min"]
            seismic_range = seismic_stats[attr]["max"] - seismic_stats[attr]["min"]

            # 计算均值比率 (避免除以零)
            if abs(well_mean) > 1e-10:
                mean_ratio = seismic_mean / well_mean
            else:
                mean_ratio = float("inf") if seismic_mean != 0 else 1.0

            # 计算标准差比率
            if abs(well_std) > 1e-10:
                std_ratio = seismic_std / well_std
            else:
                std_ratio = float("inf") if seismic_std != 0 else 1.0

            # 计算范围比率
            if abs(well_range) > 1e-10:
                range_ratio = seismic_range / well_range
            else:
                range_ratio = float("inf") if seismic_range != 0 else 1.0

            # 计算差异度量
            mean_diff = abs(mean_ratio - 1.0)

            comparison_rows.append(
                {
                    "属性": attr,
                    "井点数据均值": well_mean,
                    "地震数据均值": seismic_mean,
                    "地震/井点均值比值": mean_ratio,
                    "均值差异": mean_diff,
                    "井点数据标准差": well_std,
                    "地震数据标准差": seismic_std,
                    "标准差比值": std_ratio,
                    "井点数据范围": well_range,
                    "地震数据范围": seismic_range,
                    "范围比值": range_ratio,
                    "井点数据最小值": well_stats[attr]["min"],
                    "地震数据最小值": seismic_stats[attr]["min"],
                    "井点数据最大值": well_stats[attr]["max"],
                    "地震数据最大值": seismic_stats[attr]["max"],
                }
            )

    # 创建DataFrame并排序
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df = comparison_df.sort_values(by="均值差异", ascending=False)

    # 标记异常属性
    if exclude_negative_ratio:
        # 增加负比值检测
        anomalous_conditions = (
            (comparison_df["地震/井点均值比值"] < 0)  # 负比值
            | (abs(comparison_df["地震/井点均值比值"]) > ratio_threshold)
            | (abs(comparison_df["地震/井点均值比值"]) < 1 / ratio_threshold)
            | (abs(comparison_df["标准差比值"]) > std_ratio_threshold)
            | (abs(comparison_df["标准差比值"]) < 1 / std_ratio_threshold)
            | (abs(comparison_df["范围比值"]) > range_ratio_threshold)
            | (abs(comparison_df["范围比值"]) < 1 / range_ratio_threshold)
        )
    else:
        # 原有的检测条件
        anomalous_conditions = (
            (abs(comparison_df["地震/井点均值比值"]) > ratio_threshold)
            | (abs(comparison_df["地震/井点均值比值"]) < 1 / ratio_threshold)
            | (abs(comparison_df["标准差比值"]) > std_ratio_threshold)
            | (abs(comparison_df["标准差比值"]) < 1 / std_ratio_threshold)
            | (abs(comparison_df["范围比值"]) > range_ratio_threshold)
            | (abs(comparison_df["范围比值"]) < 1 / range_ratio_threshold)
        )

    anomalous_attrs = comparison_df[anomalous_conditions]
    good_attrs = comparison_df[~anomalous_conditions]["属性"].tolist()

    # 打印结果
    if verbose:
        print(f"\n共分析了 {len(common_attributes)} 个共同属性")
        print(f"发现 {len(anomalous_attrs)} 个异常属性")
        print(f"保留 {len(good_attrs)} 个质量良好的属性")

        if not anomalous_attrs.empty:
            print("\n异常属性及原因:")
            for _, row in anomalous_attrs.iterrows():
                reasons = []
                if row["地震/井点均值比值"] < 0:
                    reasons.append(f"均值比值为负 ({row['地震/井点均值比值']:.4f})")
                elif (
                    abs(row["地震/井点均值比值"]) > ratio_threshold
                    or abs(row["地震/井点均值比值"]) < 1 / ratio_threshold
                ):
                    reasons.append(f"均值比值异常 ({row['地震/井点均值比值']:.4f})")
                if abs(row["标准差比值"]) > std_ratio_threshold or abs(row["标准差比值"]) < 1 / std_ratio_threshold:
                    reasons.append(f"标准差比值异常 ({row['标准差比值']:.4f})")
                if abs(row["范围比值"]) > range_ratio_threshold or abs(row["范围比值"]) < 1 / range_ratio_threshold:
                    reasons.append(f"数值范围比值异常 ({row['范围比值']:.4f})")

                print(f"  - {row['属性']}: {', '.join(reasons)}")

    # 可视化部分
    if output_dir is not None:
        # 绘制异常属性的分布对比图
        if not anomalous_attrs.empty:
            anomalous_attr_list = anomalous_attrs["属性"].tolist()
            n_attrs = min(4, len(anomalous_attr_list))  # 最多显示4个
            rows = (n_attrs + 1) // 2  # 计算行数

            fig, axs = plt.subplots(rows, 2, figsize=(16, 6 * rows))
            if rows == 1 and n_attrs == 1:  # 只有一个属性的特殊情况
                axs = [axs]
            else:
                axs = axs.flatten()

            for i, attr in enumerate(anomalous_attr_list[:n_attrs]):
                # 井点数据分布
                sns.histplot(well_data[attr], color="blue", alpha=0.5, label="井点数据", ax=axs[i], kde=True)

                # 地震数据分布 (使用抽样以避免过多数据点)
                seismic_sample = seismic_data[attr].sample(min(1000, len(seismic_data)))
                sns.histplot(seismic_sample, color="red", alpha=0.5, label="地震数据(抽样)", ax=axs[i], kde=True)

                # 找到对应的行
                row = comparison_df[comparison_df["属性"] == attr].iloc[0]

                axs[i].set_title(f"异常属性: {attr}")
                axs[i].legend()

                # 在图中添加统计信息
                stats_text = (
                    f"井点: 均值={row['井点数据均值']:.4f}, 标准差={row['井点数据标准差']:.4f}\n"
                    f"地震: 均值={row['地震数据均值']:.4f}, 标准差={row['地震数据标准差']:.4f}\n"
                    f"均值比值: {row['地震/井点均值比值']:.4f}, "
                    f"标准差比值: {row['标准差比值']:.4f}, "
                    f"范围比值: {row['范围比值']:.4f}"
                )
                axs[i].text(
                    0.05,
                    0.95,
                    stats_text,
                    transform=axs[i].transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
                )

            # 隐藏多余的子图
            for i in range(n_attrs, len(axs)):
                axs[i].axis("off")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "anomalous_attributes.png"), dpi=300, bbox_inches="tight")
            plt.show()
            plt.close()

        # 创建属性质量评分图表
        plt.figure(figsize=(12, 8))
        # 创建一个复合分数来表示属性质量
        comparison_df["质量分数"] = (
            1
            / (1 + abs(comparison_df["均值差异"]))
            * 1
            / (1 + abs(np.log(comparison_df["标准差比值"] + 1e-10)))
            * 1
            / (1 + abs(np.log(comparison_df["范围比值"] + 1e-10)))
        )
        # 均值比值为负的属性质量分数设为0
        if exclude_negative_ratio:
            comparison_df.loc[comparison_df["地震/井点均值比值"] < 0, "质量分数"] = 0

        # 标准化到0-100
        max_score = comparison_df["质量分数"].max()
        comparison_df["质量分数"] = comparison_df["质量分数"] / max_score * 100

        # 按质量分数排序
        plot_df = comparison_df.sort_values("质量分数", ascending=False)

        # 添加颜色标记，异常的为红色，正常的为蓝色
        colors = ["red" if attr in anomalous_attrs["属性"].values else "blue" for attr in plot_df["属性"]]

        ax = sns.barplot(x="属性", y="质量分数", data=plot_df, palette=colors)
        plt.xticks(rotation=45, ha="right")
        plt.title("属性质量评分 (越高越好)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "attribute_quality_scores.png"), dpi=300, bbox_inches="tight")
        plt.show()
        plt.close()

    return good_attrs, anomalous_attrs, comparison_df


def preprocess_features(data, attribute_columns, missing_values=[-999], verbose=True):
    """
    预处理特征数据，包括缺失值处理、异常值替换和特征筛选

    参数:
        data (DataFrame): 包含特征的数据框
        attribute_columns (list): 需要处理的特征列名列表
        missing_values (list): 要替换为NaN的值列表，默认为[-999]
        verbose (bool): 是否打印详细信息，默认为True

    返回:
        tuple: (处理后的特征数据框, 统计信息字典)
    """
    # 提取特征
    features = data[attribute_columns].copy()

    # 替换缺失值
    for val in missing_values:
        features = features.replace(val, np.nan)

    if verbose:
        print(f"处理前特征: {features.shape}")

    # 检查缺失值情况
    missing_per_column = features.isna().sum()
    if verbose:
        print("\n每列缺失值数量:")
    missing_cols = []
    for col, missing in missing_per_column.items():
        missing_ratio = missing / len(features) * 100
        if verbose:
            print(f"  - {col}: {missing} ({missing_ratio:.2f}%)")

        # 标记缺失率较高的列
        if missing_ratio >= 89.9:
            missing_cols.append(col)

    if missing_cols:
        if verbose:
            print(f"\n删除以下全部缺失的列: {missing_cols}")
        features = features.drop(columns=missing_cols)

    # 存储每个特征的统计信息
    feature_stats = {}

    # 填充剩余列中的NaN值
    for col in features.columns:
        # 获取有效值统计量
        valid_feature_data = features[col].dropna()

        if len(valid_feature_data) > 0:
            feature_mean = valid_feature_data.mean()
            feature_std = valid_feature_data.std()
            # 确保标准差不为零，避免除零错误
            if feature_std < 1e-10:
                feature_std = 1.0
                if verbose:
                    print(f"警告: 属性 '{col}' 标准差接近零，已设为1.0")
        else:
            if verbose:
                print(f"错误: 属性 '{col}' 没有有效数据")
            feature_mean = 0.0
            feature_std = 1.0

        # 存储统计信息
        feature_stats[col] = {"mean": feature_mean, "std": feature_std}

        # 填充缺失值
        if pd.isna(feature_mean):
            if verbose:
                print(f"列 '{col}' 的均值为NaN，使用0填充")
            features[col] = features[col].fillna(0)
        else:
            features[col] = features[col].fillna(feature_mean)

    if verbose:
        print(f"\n清理并填充后的特征形状: {features.shape}")

    # 检查是否仍有NaN值
    if features.isna().any().any():
        if verbose:
            print("警告：数据中仍然存在NaN值，将它们替换为0")
        features = features.fillna(0)

    return features, feature_stats
