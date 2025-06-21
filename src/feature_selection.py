import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def select_best_features(
    well_data,
    attribute_columns,
    target_column="Thickness of facies(1: Fine sand)",
    n_features=5,
    corr_threshold=0.85,
    test_size=0.3,
    n_estimators=100,
    random_state=42,
    output_dir="output",
    verbose=True,
):
    """
    通过随机森林对所有特征进行重要性排序，然后移除高度相关的冗余特征，最终选择前n个最佳特征

    参数:
        well_data (DataFrame): 井点数据，包含地震属性和目标变量
        attribute_columns (list): 要分析的属性列名列表
        target_column (str): 目标变量列名
        n_features (int): 最终选择的特征数量
        corr_threshold (float): 相关性阈值，高于此值的属性被视为高度相关
        test_size (float): 测试集比例
        n_estimators (int): 随机森林中的树数量
        random_state (int): 随机种子
        output_dir (str): 输出目录
        verbose (bool): 是否输出详细信息

    返回:
        list: 选出的最佳特征列表
    """
    print("======== 随机森林特征重要性分析与冗余特征移除 ========")

    # 确保目标列存在
    if target_column not in well_data.columns:
        print(f"错误: 目标列 '{target_column}' 不存在于数据中")
        return []

    # 提取特征和目标变量
    X = well_data[attribute_columns].values
    y = well_data[target_column].values

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # 训练随机森林模型评估所有特征
    rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    rf.fit(X_train, y_train)

    # 评估模型性能
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if verbose:
        print(f"全特征模型性能: MSE = {mse:.4f}, R² = {r2:.4f}")

    # 获取特征重要性并创建排序字典
    importances = rf.feature_importances_
    features_importance = dict(zip(attribute_columns, importances))
    sorted_importance = sorted(features_importance.items(), key=lambda x: x[1], reverse=True)

    # 输出全部特征的重要性
    if verbose:
        print("\n全部特征重要性排序:")
        for i, (attr, imp) in enumerate(sorted_importance):
            print(f"{i + 1}. {attr}: {imp:.6f}")

    # 计算特征间的相关性
    corr_matrix = well_data[attribute_columns].corr(method="pearson")

    # 可视化特征重要性
    plt.figure(figsize=(12, max(6, len(attribute_columns) * 0.4)))
    y_pos = np.arange(len(attribute_columns))

    # 按重要性排序的特征名和值
    sorted_attrs = [attr for attr, _ in sorted_importance]
    sorted_imps = [imp for _, imp in sorted_importance]

    # 绘制条形图
    plt.barh(y_pos, sorted_imps, align="center")
    plt.yticks(y_pos, sorted_attrs)
    plt.xlabel("特征重要性")
    plt.title("所有特征的重要性排序")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "all_features_importance.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # 从最重要的特征开始，依次添加不与已选特征高度相关的特征
    selected_features = []

    for feature, importance in sorted_importance:
        # 检查当前特征与已选特征的相关性
        is_redundant = False
        for selected in selected_features:
            corr_value = abs(corr_matrix.loc[feature, selected])
            if corr_value >= corr_threshold:
                if verbose:
                    print(f"特征 '{feature}' 与已选特征 '{selected}' 相关性过高 ({corr_value:.4f})，被视为冗余")
                is_redundant = True
                break

        # 如果不是冗余特征，则添加到选择列表
        if not is_redundant:
            selected_features.append(feature)
            if verbose:
                print(f"添加特征: {feature} (重要性: {importance:.6f})")

        # 如果已经选择了足够数量的特征，则停止
        if len(selected_features) >= n_features:
            break

    # 可视化选择的特征
    selected_importances = [features_importance[feat] for feat in selected_features]
    selected_indices = [sorted_attrs.index(feat) for feat in selected_features]

    plt.figure(figsize=(12, max(6, len(selected_features) * 0.5)))
    y_pos = np.arange(len(selected_features))

    # 绘制条形图
    plt.barh(y_pos, selected_importances, align="center")
    plt.yticks(y_pos, selected_features)
    plt.xlabel("特征重要性")
    plt.title("最终选择的特征")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "selected_features_final.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # 输出最终选择的特征
    print("\n最终选择的特征列表:")
    for i, feature in enumerate(selected_features):
        print(f"{i + 1}. {feature}: {features_importance[feature]:.6f}")

    print(f"\n共选择了 {len(selected_features)} 个特征用于建模")
    print("======== 特征选择分析完成 ========")

    return selected_features
