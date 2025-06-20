import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def analyze_attribute_correlations(
    features_df,
    method="pearson",
    corr_threshold=0.85,
    output_dir="output",
    figsize=(14, 12),
):
    """
    分析属性间的相关性，将高度相关的属性分组，并可视化相关性矩阵

    参数:
        features_df (DataFrame): 包含属性列的数据框
        method (str): 相关系数计算方法，'pearson'或'spearman'，默认为'pearson'
        corr_threshold (float): 相关性阈值，高于此值的属性被视为高度相关，默认为0.85
        output_dir (str): 输出图像的目录，默认为"output"
        figsize (tuple): 图像尺寸，默认为(14, 12)

    返回:
        list: 相关属性组列表，每组包含高度相关的属性
    """
    print(f"======== 属性相关性分析开始 (方法: {method}) ========")
    print(f"属性数量: {features_df.shape[1]}")

    # 计算相关系数矩阵
    corr_matrix = features_df.corr(method=method)

    # 可视化相关系数矩阵
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 创建上三角掩码
    cmap = sns.diverging_palette(230, 20, as_cmap=True)  # 使用发散色调色板

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=0.5,
        annot=False,  # 不显示具体数值，避免过于拥挤
        cbar_kws={"shrink": 0.8},
    )

    plt.title(f"属性相关性矩阵 ({method}相关系数)", fontsize=16)
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"attribute_correlation_{method}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # 提取相关属性组 - 修复后的代码
    attrs = list(corr_matrix.columns)
    n_attrs = len(attrs)
    correlated_groups = []
    processed_attrs = set()

    # 第一步：找出所有高相关性的属性对
    for i in range(n_attrs):
        attr1 = attrs[i]
        if attr1 in processed_attrs:
            continue

        # 开始一个新组
        current_group = [attr1]
        processed_attrs.add(attr1)

        # 查找所有与attr1高度相关的属性
        for j in range(n_attrs):
            attr2 = attrs[j]
            if attr1 != attr2 and attr2 not in processed_attrs:
                corr_value = abs(corr_matrix.loc[attr1, attr2])
                if corr_value >= corr_threshold:
                    current_group.append(attr2)
                    processed_attrs.add(attr2)

        # 添加当前组到结果
        if len(current_group) >= 1:  # 即使只有一个属性也添加
            correlated_groups.append(set(current_group))

    # 确保所有属性都被处理
    remaining_attrs = set(attrs) - processed_attrs
    for attr in remaining_attrs:
        correlated_groups.append({attr})

    # 输出相关属性组
    print("\n高度相关的属性组 (相关系数阈值 >= {}):".format(corr_threshold))
    multi_attr_groups = [g for g in correlated_groups if len(g) > 1]
    for i, group in enumerate(multi_attr_groups):
        print(f"\n组 {i + 1} (包含 {len(group)} 个属性):")
        for attr in group:
            print(f"  - {attr}")

    # 输出独立属性
    independent_attrs = [list(g)[0] for g in correlated_groups if len(g) == 1]
    print(f"\n独立属性 ({len(independent_attrs)} 个):")
    for attr in independent_attrs:
        print(f"  - {attr}")

    # 计算组内平均相关系数 - 修复计算方法
    if len(multi_attr_groups) > 0:
        print("\n各组内部平均相关系数:")
        for i, group in enumerate(multi_attr_groups):
            if len(group) > 1:
                group_list = list(group)
                group_corr = corr_matrix.loc[group_list, group_list].values

                # 计算上三角矩阵(不含对角线)的平均值
                tri_mask = np.triu(np.ones(group_corr.shape), k=1).astype(bool)
                mean_corr = np.abs(group_corr[tri_mask]).mean()

                print(f"  - 组 {i + 1}: {mean_corr:.4f}")

    print(f"======== 属性相关性分析完成 ========")

    return correlated_groups


def analyze_rf_importance_by_group(
    well_data,
    attribute_groups,
    target_column="Thickness of LITHOLOGIES(1: sand)",
    top_n=1,
    test_size=0.3,
    n_estimators=100,
    random_state=42,
    output_dir="output",
    verbose=True,
):
    """
    对每组属性进行随机森林重要性分析，选择每组中最重要的特征

    参数:
        well_data (DataFrame): 井点数据，包含地震属性和目标变量
        attribute_groups (list): 属性组列表，每组是一个包含相关属性的集合
        target_column (str): 目标变量列名，默认为"Thickness of LITHOLOGIES(1: sand)"
        top_n (int): 从每组中选择的最重要特征数量，默认为1
        test_size (float): 测试集比例，默认为0.3
        n_estimators (int): 随机森林中的树数量，默认为100
        random_state (int): 随机种子，默认为42
        output_dir (str): 输出目录，默认为"output"
        verbose (bool): 是否输出详细信息，默认为True

    返回:
        list: 所有组中选出的最重要特征列表
    """
    print("======== 按组进行随机森林特征重要性分析 ========")

    # 确保目标列存在
    if target_column not in well_data.columns:
        print(f"错误: 目标列 '{target_column}' 不存在于数据中")
        return []

    # 准备目标变量
    y = well_data[target_column].values

    # 初始化结果存储
    selected_features = []
    all_importances = {}

    # 对每组属性进行分析
    for i, group in enumerate(attribute_groups):
        group_list = list(group)

        # 如果组中只有一个属性，直接添加到结果中
        if len(group_list) == 1:
            selected_features.append(group_list[0])
            if verbose:
                print(f"\n组 {i + 1} 只有一个属性: {group_list[0]}，直接添加到选择列表")
            continue

        # 打印当前组信息
        if verbose:
            print(f"\n分析组 {i + 1} (包含 {len(group_list)} 个属性):")
            for attr in group_list:
                print(f"  - {attr}")

        # 提取当前组的特征
        X_group = well_data[group_list].values

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_group, y, test_size=test_size, random_state=random_state
        )

        # 训练随机森林模型
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,  # 使用所有CPU核心
        )
        rf.fit(X_train, y_train)

        # 评估模型性能
        y_pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        if verbose:
            print(f"  组 {i + 1} 模型性能: MSE = {mse:.4f}, R² = {r2:.4f}")

        # 获取特征重要性
        importances = rf.feature_importances_
        importance_dict = dict(zip(group_list, importances))

        # 按重要性排序
        sorted_importances = sorted(
            importance_dict.items(), key=lambda x: x[1], reverse=True
        )

        # 保存每个属性的重要性
        for attr, imp in sorted_importances:
            all_importances[attr] = imp

        # 输出该组属性的重要性
        if verbose:
            print(f"  组 {i + 1} 特征重要性:")
            for attr, imp in sorted_importances:
                print(f"    - {attr}: {imp:.6f}")

        # 选择 top_n 个重要性最高的特征
        top_features = [
            attr
            for attr, _ in sorted_importances[: min(top_n, len(sorted_importances))]
        ]
        selected_features.extend(top_features)

        if verbose:
            print(f"  从组 {i + 1} 中选择的特征: {', '.join(top_features)}")

        # 可视化特征重要性
        plt.figure(figsize=(10, max(6, len(group_list) * 0.5)))
        y_pos = np.arange(len(group_list))

        # 排序后的重要性和属性名
        sorted_attrs = [attr for attr, _ in sorted_importances]
        sorted_imps = [imp for _, imp in sorted_importances]

        # 绘制条形图
        bars = plt.barh(y_pos, sorted_imps, align="center")

        # 为top_n特征着色
        for j in range(min(top_n, len(bars))):
            bars[j].set_color("red")

        plt.yticks(y_pos, sorted_attrs)
        plt.xlabel("特征重要性")
        plt.title(f"组 {i + 1} 的特征重要性")
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"group_{i + 1}_feature_importance.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    # 可视化所有选择的特征及其重要性
    selected_importances = {
        feat: all_importances.get(feat, 0) for feat in selected_features
    }
    sorted_selected = sorted(
        selected_importances.items(), key=lambda x: x[1], reverse=True
    )

    plt.figure(figsize=(12, max(6, len(selected_features) * 0.4)))
    y_pos = np.arange(len(selected_features))

    # 排序后的重要性和属性名
    sorted_selected_attrs = [attr for attr, _ in sorted_selected]
    sorted_selected_imps = [imp for _, imp in sorted_selected]

    # 绘制条形图
    plt.barh(y_pos, sorted_selected_imps, align="center")
    plt.yticks(y_pos, sorted_selected_attrs)
    plt.xlabel("特征重要性")
    plt.title("所有选择特征的重要性")
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "selected_features_importance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    # 输出最终选择的特征
    print("\n最终选择的特征列表:")
    for i, feature in enumerate(sorted_selected_attrs):
        print(f"{i + 1}. {feature}: {selected_importances[feature]:.6f}")

    print(f"\n共选择了 {len(selected_features)} 个特征用于建模")
    print("======== 随机森林特征重要性分析完成 ========")

    return sorted_selected_attrs
