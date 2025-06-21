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
    n_runs=10,  # 运行随机森林的次数
    max_depth=4,  # 控制树的深度，避免过拟合
    random_state=42,
    output_dir="output",
    verbose=True,
):
    """
    通过多次运行随机森林对特征进行重要性排序，选择排名稳定的特征，并移除高度相关的冗余特征

    参数:
        well_data (DataFrame): 井点数据，包含地震属性和目标变量
        attribute_columns (list): 要分析的属性列名列表
        target_column (str): 目标变量列名
        n_features (int): 最终选择的特征数量
        corr_threshold (float): 相关性阈值，高于此值的属性被视为高度相关
        test_size (float): 测试集比例
        n_estimators (int): 随机森林中的树数量
        n_runs (int): 运行随机森林的次数，用于评估特征重要性的稳定性
        max_depth (int): 随机森林中树的最大深度
        random_state (int): 随机种子
        output_dir (str): 输出目录
        verbose (bool): 是否输出详细信息

    返回:
        list: 选出的最佳特征列表
    """
    print("======== 多次运行随机森林特征重要性分析 ========")

    # 确保目标列存在
    if target_column not in well_data.columns:
        print(f"错误: 目标列 '{target_column}' 不存在于数据中")
        return []

    # 初始化结果存储
    importance_ranks = {attr: [] for attr in attribute_columns}
    importance_values = {attr: [] for attr in attribute_columns}

    # 多次运行随机森林
    for run in range(n_runs):
        if verbose:
            print(f"\n运行 {run + 1}/{n_runs}")

        # 生成不同的随机种子
        current_seed = random_state + run

        # 提取特征和目标变量
        X = well_data[attribute_columns].values
        y = well_data[target_column].values

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=current_seed)

        # 训练随机森林模型评估所有特征
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,  # 限制树的深度
            random_state=current_seed,
            n_jobs=-1,
        )
        rf.fit(X_train, y_train)

        # 评估模型性能
        y_pred = rf.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        if verbose:
            print(f"模型性能: MSE = {mse:.4f}, R² = {r2:.4f}")

        # 获取特征重要性
        importances = rf.feature_importances_
        features_importance = dict(zip(attribute_columns, importances))

        # 排序并记录每个特征的排名和重要性值
        sorted_importance = sorted(features_importance.items(), key=lambda x: x[1], reverse=True)

        for rank, (attr, imp) in enumerate(sorted_importance):
            importance_ranks[attr].append(rank + 1)  # 排名从1开始
            importance_values[attr].append(imp)

    # 计算平均排名和平均重要性
    avg_ranks = {attr: np.mean(ranks) for attr, ranks in importance_ranks.items()}
    avg_importance = {attr: np.mean(values) for attr, values in importance_values.items()}
    rank_std = {attr: np.std(ranks) for attr, ranks in importance_ranks.items()}

    # 按平均排名排序
    sorted_by_avg_rank = sorted(avg_ranks.items(), key=lambda x: x[1])

    if verbose:
        print("\n特征平均排名和稳定性:")
        for attr, avg_rank in sorted_by_avg_rank:
            print(
                f"{attr}: 平均排名 = {avg_rank:.2f}, 排名标准差 = {rank_std[attr]:.2f}, 平均重要性 = {avg_importance[attr]:.6f}"
            )

    # 可视化平均特征重要性和排名稳定性
    plt.figure(figsize=(14, max(8, len(attribute_columns) * 0.5)))

    # 按平均排名排序
    sorted_attrs = [attr for attr, _ in sorted_by_avg_rank]
    avg_imps = [avg_importance[attr] for attr in sorted_attrs]
    stds = [rank_std[attr] for attr in sorted_attrs]

    y_pos = np.arange(len(sorted_attrs))

    # 绘制条形图
    bars = plt.barh(y_pos, avg_imps, align="center", alpha=0.7)

    # 添加排名稳定性标注
    for i, (bar, std) in enumerate(zip(bars, stds)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2, f"σ={std:.2f}", va="center", fontsize=9)

    plt.yticks(y_pos, sorted_attrs)
    plt.xlabel("平均特征重要性")
    plt.title("多次运行后的特征重要性和排名稳定性(σ)")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance_stability.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # 计算特征间的相关性
    corr_matrix = well_data[attribute_columns].corr(method="pearson")

    # 从平均排名最高的特征开始，依次添加不与已选特征高度相关的特征
    selected_features = []

    for attr, _ in sorted_by_avg_rank:
        # 检查当前特征与已选特征的相关性
        is_redundant = False
        for selected in selected_features:
            corr_value = abs(corr_matrix.loc[attr, selected])
            if corr_value >= corr_threshold:
                if verbose:
                    print(f"特征 '{attr}' 与已选特征 '{selected}' 相关性过高 ({corr_value:.4f})，被视为冗余")
                is_redundant = True
                break

        # 如果不是冗余特征，则添加到选择列表
        if not is_redundant:
            selected_features.append(attr)
            if verbose:
                print(f"添加特征: {attr} (平均排名: {avg_ranks[attr]:.2f}, 排名稳定性: {rank_std[attr]:.2f})")

        # 如果已经选择了足够数量的特征，则停止
        if len(selected_features) >= n_features:
            break

    # 可视化最终选择的特征
    plt.figure(figsize=(12, max(6, len(selected_features) * 0.6)))
    y_pos = np.arange(len(selected_features))

    selected_importances = [avg_importance[feat] for feat in selected_features]
    selected_stds = [rank_std[feat] for feat in selected_features]

    # 绘制条形图
    bars = plt.barh(y_pos, selected_importances, align="center")

    # 添加排名稳定性标注
    for i, (bar, std) in enumerate(zip(bars, selected_stds)):
        plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2, f"σ={std:.2f}", va="center", fontsize=9)

    plt.yticks(y_pos, selected_features)
    plt.xlabel("平均特征重要性")
    plt.title("最终选择的特征及其排名稳定性")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "selected_features_final.png"), dpi=300, bbox_inches="tight")
    plt.show()

    # 输出最终选择的特征
    print("\n最终选择的特征列表:")
    for i, feature in enumerate(selected_features):
        print(f"{i + 1}. {feature}: 平均排名 = {avg_ranks[feature]:.2f}, 排名稳定性 = {rank_std[feature]:.2f}")

    print(f"\n共选择了 {len(selected_features)} 个特征用于建模")
    print("======== 特征选择分析完成 ========")

    return selected_features
