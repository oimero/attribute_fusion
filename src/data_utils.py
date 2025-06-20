import numpy as np
import pandas as pd


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
        # 跳过ATTRIBUTES行和下一行的1,N
        for i in range(attr_start + 2, attr_end):
            line = lines[i].strip()
            # 直接查找最后一个英文逗号，之后的内容即为属性名
            last_comma_index = line.rfind(",")
            if last_comma_index != -1:
                attr_name = line[last_comma_index + 1 :].strip()
                attributes.append(attr_name)

    print(f"识别到 {len(attributes)} 个属性:")
    for attr in attributes:
        print(f"  - {attr}")

    return attributes, attr_end


def parse_petrel_file(file_path):
    """
    解析Petrel格式的地震属性文件

    参数:
        file_path (str): Petrel文件的路径

    返回:
        pandas.DataFrame: 包含解析后数据的DataFrame
    """
    print(f"正在解析文件: {file_path}")

    # 步骤1: 识别属性和确定跳过的行数
    attributes, skip_rows = identify_attributes(file_path)

    # 读取文件内容以分析数据结构
    with open(file_path, "r") as file:
        lines = file.readlines()

    # 步骤2: 解析数据列数
    if skip_rows + 1 < len(lines):
        first_data_line = lines[skip_rows + 1].strip()
        data_columns = first_data_line.split()
        num_columns = len(data_columns)
        print(f"解析到数据有 {num_columns} 列")
    else:
        print("未找到数据行")
        return None

    # 步骤3: 创建列名
    column_names = ["X", "Y", "Z"]

    # 添加中间的占位符列名
    placeholder_count = num_columns - 3 - len(attributes)
    for i in range(placeholder_count):
        column_names.append(f"placeholder{i + 1}")

    # 添加属性列名
    column_names.extend(attributes)

    print(f"总列数: {num_columns}, 其中:")
    print(f"  - 3 列为坐标 (X, Y, Z)")
    print(f"  - {placeholder_count} 列为占位符")
    print(f"  - {len(attributes)} 列为属性")

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
        return df
    except Exception as e:
        print(f"读取数据时出错: {str(e)}")
        return None


def preprocess_features(data, attribute_columns, missing_values=[-999], verbose=True):
    """
    预处理特征数据，包括缺失值处理、异常值替换和特征筛选

    参数:
        data (DataFrame): 包含特征的数据框
        attribute_columns (list): 需要处理的特征列名列表
        missing_values (list): 要替换为NaN的值列表，默认为[-999]
        verbose (bool): 是否打印详细信息，默认为True

    返回:
        DataFrame: 处理后的特征数据框
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

    # 填充剩余列中的NaN值
    # 对于每一列，如果均值是NaN，则使用0填充
    for col in features.columns:
        if pd.isna(features[col].mean()):
            if verbose:
                print(f"列 '{col}' 的均值为NaN，使用0填充")
            features[col] = features[col].fillna(0)
        else:
            features[col] = features[col].fillna(features[col].mean())

    if verbose:
        print(f"\n清理并填充后的特征形状: {features.shape}")

    # 检查是否仍有NaN值
    if features.isna().any().any():
        if verbose:
            print("警告：数据中仍然存在NaN值，将它们替换为0")
        features = features.fillna(0)

    return features
