import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree


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
    header_field_count = 0

    for line in lines:
        if line.startswith("# Field"):
            parts = line.split(":")
            if len(parts) >= 2:
                field_name = parts[1].strip()
                column_meanings.append(field_name)
                header_field_count += 1

    print(f"识别到的列含义: {column_meanings}")
    print(f"文件头定义的基础列数: {header_field_count}")

    # 步骤2: 构建列名
    # 使用文件头中定义的基础列名
    if column_meanings:
        column_names = column_meanings.copy()
    else:
        # 如果没有找到文件头定义，使用默认值
        column_names = ["X", "Y", "Z", "column", "row"]
        header_field_count = 5

    # 添加属性列名
    column_names.extend(attributes)

    # 确认实际数据列数
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

        if len(column_names) > num_columns:
            # 如果列名太多，截断属性列名
            excess_count = len(column_names) - num_columns
            print(f"截断属性列名，删除最后 {excess_count} 个属性")
            column_names = column_names[:num_columns]

        else:
            # 如果列名太少，在基础列和属性列之间添加占位符
            missing_count = num_columns - len(column_names)
            print(f"需要添加 {missing_count} 个占位符列名")

            # 在基础列（如X,Y,Z,column,row）之后、属性列之前插入占位符
            base_columns = column_names[:header_field_count]  # 基础列
            attr_columns = column_names[header_field_count:]  # 属性列

            # 生成占位符列名
            placeholder_columns = []
            for i in range(missing_count):
                placeholder_columns.append(f"Unknown_Column_{i + 1}")

            # 重新组合列名：基础列 + 占位符 + 属性列
            column_names = base_columns + placeholder_columns + attr_columns

            print(f"在基础列后添加占位符列名: {placeholder_columns}")
            print(f"最终列名列表长度: {len(column_names)}")

    # 验证最终列名数量
    if len(column_names) != num_columns:
        print(f"错误: 调整后的列名数量({len(column_names)})仍与数据列数({num_columns})不匹配!")
        return None

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

        # 只显示关键列的统计信息，避免输出过多占位符列
        key_columns = ["X", "Y", "Z"] + [col for col in df.columns if col in attributes][:5]  # 显示前5个属性
        available_key_columns = [col for col in key_columns if col in df.columns]

        print(f"\n关键列统计信息 (显示{len(available_key_columns)}列):")
        print(df[available_key_columns].describe().T[["min", "max", "mean", "std"]])

        return df
    except Exception as e:
        print(f"读取数据时出错: {str(e)}")
        return None


def preprocess_features(
    data,
    attribute_columns,
    missing_values=[-999],
    missing_threshold=0.6,
    outlier_method="iqr",
    outlier_threshold=1.5,
    verbose=True,
):
    """
    预处理特征数据，包括缺失值处理、异常值替换和特征筛选

    参数:
        data (DataFrame): 包含特征的数据框
        attribute_columns (list): 需要处理的特征列名列表
        missing_values (list): 要替换为NaN的值列表，默认为[-999]
        missing_threshold (float): 缺失值占比阈值，超过此值的列将被删除，默认为0.6 (60%)
        outlier_method (str): 离群值检测方法，可选 'iqr'(四分位距) 或 'zscore'(标准分数)
        outlier_threshold (float): 离群值判定阈值，默认为1.5 (IQR方法) 或 3.0 (Z-score方法)
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
    missing_ratio_per_column = missing_per_column / len(features)

    if verbose:
        print("\n每列缺失值情况:")

    # 根据缺失值阈值筛选列
    high_missing_cols = []
    for col, missing_ratio in missing_ratio_per_column.items():
        if verbose:
            print(f"  - {col}: {missing_per_column[col]} ({missing_ratio * 100:.2f}%)")

        if missing_ratio >= missing_threshold:
            high_missing_cols.append(col)

    if high_missing_cols:
        if verbose:
            print(f"\n删除以下缺失值比例 >= {missing_threshold * 100}% 的列: {high_missing_cols}")
        features = features.drop(columns=high_missing_cols)

    # 存储每个特征的统计信息
    feature_stats = {}

    # 处理剩余列中的缺失值和离群值
    for col in features.columns:
        # 获取有效值
        valid_data = features[col].dropna()

        if len(valid_data) == 0:
            if verbose:
                print(f"警告: 属性 '{col}' 没有有效数据，将使用0填充")
            features[col] = features[col].fillna(0)
            feature_stats[col] = {"mean": 0.0, "std": 1.0, "median": 0.0, "q1": 0.0, "q3": 0.0}
            continue

        # 检测并处理离群值
        if outlier_method == "iqr":
            # 使用IQR方法识别离群值
            q1 = valid_data.quantile(0.25)
            q3 = valid_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - outlier_threshold * iqr
            upper_bound = q3 + outlier_threshold * iqr

            # 筛选非离群数据
            clean_data = valid_data[(valid_data >= lower_bound) & (valid_data <= upper_bound)]

            # 存储统计信息
            feature_stats[col] = {
                "mean": clean_data.mean(),
                "std": clean_data.std(),
                "median": clean_data.median(),
                "q1": q1,
                "q3": q3,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }

        elif outlier_method == "zscore":
            # 使用Z-score方法识别离群值
            mean = valid_data.mean()
            std = valid_data.std()

            if std < 1e-10:
                std = 1.0
                if verbose:
                    print(f"警告: 属性 '{col}' 标准差接近零，已设为1.0")

            z_scores = np.abs((valid_data - mean) / std)
            clean_data = valid_data[z_scores <= outlier_threshold]

            # 存储统计信息
            feature_stats[col] = {
                "mean": clean_data.mean(),
                "std": clean_data.std(),
                "median": clean_data.median(),
                "mean_with_outliers": mean,
                "std_with_outliers": std,
            }

        else:
            raise ValueError(f"不支持的离群值检测方法: {outlier_method}，请使用 'iqr' 或 'zscore'")

        # 检查是否有缺失值需要填充
        if features[col].isna().any():
            # 使用清理后的数据计算填充值
            fill_value = clean_data.mean()

            # 检查填充值是否有效
            if pd.isna(fill_value):
                if verbose:
                    print(f"警告: 属性 '{col}' 的计算填充值为NaN，将使用原始数据的中位数")
                fill_value = valid_data.median()
                if pd.isna(fill_value):
                    if verbose:
                        print(f"警告: 属性 '{col}' 的中位数仍为NaN，将使用0")
                    fill_value = 0

            # 填充缺失值
            features[col] = features[col].fillna(fill_value)

            # 只打印有缺失值的属性的填充信息
            if verbose:
                print(f"  - 属性 '{col}' 的填充值为 {fill_value:.4f}")

    if verbose:
        print(f"\n清理并填充后的特征形状: {features.shape}")

    # 最终检查是否仍有NaN值
    if features.isna().any().any():
        if verbose:
            print("警告：数据中仍然存在NaN值，将它们替换为0")
        features = features.fillna(0)

    return features, feature_stats


def filter_outlier_wells(well_data, method="iqr", distance_threshold=None):
    """
    筛选并剔除离群井

    参数:
    well_data: 井点数据DataFrame
    method: 'iqr'使用箱线图方法，'distance'使用距离方法
    distance_threshold: 使用distance方法时的距离阈值

    返回:
    filtered_well_data: 过滤后的井点数据
    """
    if method == "iqr":
        # 使用箱线图方法 (IQR = Q3 - Q1)
        Q1_x = well_data["X"].quantile(0.25)
        Q3_x = well_data["X"].quantile(0.75)
        IQR_x = Q3_x - Q1_x

        Q1_y = well_data["Y"].quantile(0.25)
        Q3_y = well_data["Y"].quantile(0.75)
        IQR_y = Q3_y - Q1_y

        # 定义异常值边界 (通常是Q1-1.5*IQR和Q3+1.5*IQR)
        lower_bound_x = Q1_x - 1.5 * IQR_x
        upper_bound_x = Q3_x + 1.5 * IQR_x
        lower_bound_y = Q1_y - 1.5 * IQR_y
        upper_bound_y = Q3_y + 1.5 * IQR_y

        # 筛选正常范围内的井点
        filtered_well_data = well_data[
            (well_data["X"] >= lower_bound_x)
            & (well_data["X"] <= upper_bound_x)
            & (well_data["Y"] >= lower_bound_y)
            & (well_data["Y"] <= upper_bound_y)
        ]

    elif method == "distance":
        if distance_threshold is None:
            raise ValueError("使用distance方法时必须提供distance_threshold参数")

        # 计算井点的中心位置
        center_x = well_data["X"].mean()
        center_y = well_data["Y"].mean()

        # 计算每个井点到中心的距离
        well_data["distance_to_center"] = np.sqrt((well_data["X"] - center_x) ** 2 + (well_data["Y"] - center_y) ** 2)

        # 筛选距离中心不超过阈值的井点
        filtered_well_data = well_data[well_data["distance_to_center"] <= distance_threshold]
        filtered_well_data = filtered_well_data.drop(columns=["distance_to_center"])

    else:
        raise ValueError("method参数必须是'iqr'或'distance'")

    return filtered_well_data.reset_index(drop=True)


def extract_seismic_attributes_at_location(seismic_data, x, y, max_distance=None, num_points=None):
    """
    提取指定位置附近的地震属性平均值

    参数:
    seismic_data: 地震数据，包含X、Y坐标和属性值
    x, y: 目标位置的X、Y坐标
    max_distance: 最大距离范围，超过此距离的点不会被考虑
    num_points: 最多使用的点数量

    返回:
    attributes_dict: 包含平均属性值的字典
    """
    # 计算所有地震点到目标点的距离
    distances = np.sqrt((seismic_data["X"] - x) ** 2 + (seismic_data["Y"] - y) ** 2)

    # 将距离添加到数据中
    temp_data = seismic_data.copy()
    temp_data["distance"] = distances

    # 根据距离排序
    temp_data = temp_data.sort_values("distance")

    # 应用最大距离过滤
    if max_distance is not None:
        temp_data = temp_data[temp_data["distance"] <= max_distance]

    # 应用点数量限制
    if num_points is not None and len(temp_data) > num_points:
        temp_data = temp_data.iloc[:num_points]

    # 如果没有符合条件的点，返回空字典
    if len(temp_data) == 0:
        print(f"警告: 在坐标({x}, {y})附近没有找到符合条件的地震点")
        return {}

    # 获取所有属性列名（排除X、Y和distance列）
    attribute_cols = [col for col in temp_data.columns if col not in ["X", "Y", "distance", "INLINE", "XLINE", "Z"]]

    # 计算每个属性的平均值
    attributes_dict = {}
    for attr in attribute_cols:
        attributes_dict[attr] = temp_data[attr].mean()

    # 返回包含使用的点数量信息
    attributes_dict["num_points_used"] = len(temp_data)
    attributes_dict["avg_distance"] = temp_data["distance"].mean()

    return attributes_dict


def extract_seismic_attributes_for_wells(well_data, seismic_data, max_distance=None, num_points=None):
    """
    为所有井点提取地震属性

    参数:
    well_data: 井点数据DataFrame
    seismic_data: 地震数据DataFrame
    max_distance: 最大距离范围
    num_points: 每个井点使用的最多点数量

    返回:
    well_data_with_attributes: 包含地震属性的井点数据
    """
    # 创建结果DataFrame的副本
    well_data_with_attributes = well_data.copy()

    # 遍历每个井点
    for idx, well in well_data.iterrows():
        # 提取该井点处的地震属性
        attributes = extract_seismic_attributes_at_location(
            seismic_data, well["X"], well["Y"], max_distance, num_points
        )

        # 如果找到了属性，则添加到结果中
        if attributes:
            for attr_name, attr_value in attributes.items():
                well_data_with_attributes.loc[idx, attr_name] = attr_value

    return well_data_with_attributes


def filter_seismic_by_wells(
    seismic_data, well_data, expansion_factor=1.5, plot=True, output_dir=None, figsize=(15, 10)
):
    """
    根据井点分布范围筛选地震数据

    参数:
        seismic_data (DataFrame): 地震数据
        well_data (DataFrame): 井点数据
        expansion_factor (float): 范围扩展比例，默认1.5（扩展50%）
        plot (bool): 是否绘制可视化图形
        output_dir (str): 输出目录，如果为None则不保存图片
        figsize (tuple): 图形大小

    返回:
        DataFrame: 筛选后的地震数据
        dict: 区域边界信息
    """
    # 获取井点数据的X、Y范围
    well_x_min = well_data["X"].min()
    well_x_max = well_data["X"].max()
    well_y_min = well_data["Y"].min()
    well_y_max = well_data["Y"].max()

    # 计算中心点坐标（可用于后续处理）
    well_x_center = (well_x_min + well_x_max) / 2
    well_y_center = (well_y_min + well_y_max) / 2

    # 打印井点区域范围
    print(f"井点数据X轴范围: {well_x_min:.2f} 到 {well_x_max:.2f}")
    print(f"井点数据Y轴范围: {well_y_min:.2f} 到 {well_y_max:.2f}")

    # 计算扩展边界
    x_padding = (well_x_max - well_x_min) * (expansion_factor - 1) / 2
    y_padding = (well_y_max - well_y_min) * (expansion_factor - 1) / 2

    # 应用扩展后的范围
    area_x_min = well_x_min - x_padding
    area_x_max = well_x_max + x_padding
    area_y_min = well_y_min - y_padding
    area_y_max = well_y_max + y_padding

    # 筛选出井点范围内的地震数据
    filtered_data = seismic_data[
        (seismic_data["X"] >= area_x_min)
        & (seismic_data["X"] <= area_x_max)
        & (seismic_data["Y"] >= area_y_min)
        & (seismic_data["Y"] <= area_y_max)
    ].copy()

    # 统计过滤前后的数据量
    original_size = len(seismic_data)
    filtered_size = len(filtered_data)
    reduction_percent = (1 - filtered_size / original_size) * 100

    print(f"原始地震数据点数: {original_size}")
    print(f"缩小范围后的地震数据点数: {filtered_size}")
    print(f"数据量减少了: {reduction_percent:.2f}%")

    # 记录区域边界信息
    area_bounds = {
        "x_min": area_x_min,
        "x_max": area_x_max,
        "y_min": area_y_min,
        "y_max": area_y_max,
        "x_center": well_x_center,
        "y_center": well_y_center,
        "expansion_factor": expansion_factor,
    }

    # 可视化原始数据与筛选后的数据分布
    if plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize)

        # 绘制地震数据点（使用抽样以避免过多点导致图像渲染缓慢）
        sample_ratio = min(1.0, 5000 / len(seismic_data))
        seismic_sample = seismic_data.sample(frac=sample_ratio)
        plt.scatter(
            seismic_sample["X"], seismic_sample["Y"], color="lightgray", alpha=0.3, s=10, label="原始地震数据(抽样)"
        )

        # 绘制筛选后的地震数据
        filtered_sample_ratio = min(1.0, 3000 / len(filtered_data))
        filtered_sample = filtered_data.sample(frac=filtered_sample_ratio)
        plt.scatter(
            filtered_sample["X"], filtered_sample["Y"], color="blue", alpha=0.5, s=15, label="筛选后的地震数据(抽样)"
        )

        # 绘制井点位置
        plt.scatter(well_data["X"], well_data["Y"], color="red", s=80, marker="^", label="井点位置")

        # 绘制筛选边界框
        plt.axvline(x=area_x_min, color="red", linestyle="--", alpha=0.8)
        plt.axvline(x=area_x_max, color="red", linestyle="--", alpha=0.8)
        plt.axhline(y=area_y_min, color="red", linestyle="--", alpha=0.8)
        plt.axhline(y=area_y_max, color="red", linestyle="--", alpha=0.8)

        # 添加标题和图例
        plt.title("地震数据与井点分布", fontsize=16)
        plt.xlabel("X坐标", fontsize=14)
        plt.ylabel("Y坐标", fontsize=14)
        plt.legend(loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.7)

        # 保存图片（如果指定了输出目录）
        if output_dir:
            import os

            plt.savefig(os.path.join(output_dir, "seismic_well_distribution.png"), dpi=300, bbox_inches="tight")

        plt.show()

    return filtered_data, area_bounds


def extract_uniform_seismic_samples(seismic_data, n_rows=15, n_cols=15, random_seed=42, area_bounds=None):
    """
    在地震数据中等间距地提取样本点（用于后续虚拟井点生成）

    参数:
        seismic_data (DataFrame): 包含X和Y坐标的地震数据
        n_rows (int): 采样网格的行数
        n_cols (int): 采样网格的列数
        random_seed (int): 随机数种子，用于可重复性
        area_bounds (dict, optional): 指定采样区域的边界，格式为
                                     {'x_min': float, 'x_max': float,
                                      'y_min': float, 'y_max': float}
                                     如果为None，则使用整个数据集的范围

    返回:
        DataFrame: 包含均匀分布的样本点的X、Y坐标和对应的地震属性
    """

    np.random.seed(random_seed)

    # 如果未指定边界，使用数据的实际范围
    if area_bounds is None:
        x_min, x_max = seismic_data["X"].min(), seismic_data["X"].max()
        y_min, y_max = seismic_data["Y"].min(), seismic_data["Y"].max()
    else:
        x_min, x_max = area_bounds["x_min"], area_bounds["x_max"]
        y_min, y_max = area_bounds["y_min"], area_bounds["y_max"]

    print(f"采样区域范围: X=[{x_min:.2f}, {x_max:.2f}], Y=[{y_min:.2f}, {y_max:.2f}]")

    # 检查地震数据是否覆盖了指定区域
    in_area = (
        (seismic_data["X"] >= x_min)
        & (seismic_data["X"] <= x_max)
        & (seismic_data["Y"] >= y_min)
        & (seismic_data["Y"] <= y_max)
    )

    seismic_in_area = seismic_data[in_area]
    print(
        f"指定区域内的地震数据点数: {len(seismic_in_area)} / {len(seismic_data)} ({len(seismic_in_area) / len(seismic_data) * 100:.2f}%)"
    )

    if len(seismic_in_area) == 0:
        print("警告: 指定区域内没有地震数据点!")
        # 可以选择在这里返回空DataFrame或修改区域边界

    # 创建均匀网格 - 直接使用指定的行数和列数
    x_points = np.linspace(x_min, x_max, n_cols)
    y_points = np.linspace(y_min, y_max, n_rows)

    # 生成网格点
    xx, yy = np.meshgrid(x_points, y_points)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    total_points = n_rows * n_cols
    print(f"生成了 {total_points} 个等间距采样点 ({n_rows}行 x {n_cols}列)")

    # 创建DataFrame存储采样点
    seismic_samples = pd.DataFrame(grid_points, columns=["X", "Y"])

    # 构建KD树用于快速最近邻搜索
    seismic_points = seismic_data[["X", "Y"]].values
    tree = cKDTree(seismic_points)

    # 查找每个采样点的最近地震点
    distances, indices = tree.query(seismic_samples[["X", "Y"]].values, k=1)

    # 打印距离统计信息
    print(
        f"最近距离统计: 最小={distances.min():.2f}, 最大={distances.max():.2f}, 平均={distances.mean():.2f}, 中位数={np.median(distances):.2f}"
    )

    # 获取每个采样点对应的地震属性 - 直接获取，不筛选NaN
    for col in seismic_data.columns:
        if col not in ["X", "Y"]:
            seismic_samples[col] = seismic_data.iloc[indices][col].values

    # 添加样本编号 - 包含行列信息
    sample_ids = []
    for i in range(n_rows):
        for j in range(n_cols):
            sample_ids.append(f"S{i + 1:02d}{j + 1:02d}")  # 例如S0101表示第1行第1列

    seismic_samples["Sample_ID"] = sample_ids

    print(f"最终返回 {len(seismic_samples)} 个采样点")

    return seismic_samples


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
