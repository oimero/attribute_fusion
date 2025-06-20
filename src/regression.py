import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from src.visualization import (
    export_to_petrel_format,
    visualize_model_results,
    visualize_predictions,
)


def build_svr_model(
    data,
    selected_features,
    target_column="Thickness of LITHOLOGIES(1: sand)",
    use_cluster_probs=False,  # 是否使用聚类概率
    use_onehot=False,  # 是否使用One-Hot编码的聚类特征
    test_size=0.3,
    random_state=42,
    output_dir="output",
    filename_prefix=None,  # 添加文件名前缀参数
    threshold_zero=0.1,  # 小于此值视为0
    n_splits=5,
    model_params=None,  # 可选的SVR参数
    verbose=True,
):
    """
    构建SVR回归模型，支持不同的特征组合

    参数:
        data (DataFrame): 包含特征和目标变量的数据框
        selected_features (list): 选定的基础特征列表
        target_column (str): 目标变量列名
        use_cluster_probs (bool): 是否包含聚类概率作为附加特征
        use_onehot (bool): 是否使用One-Hot编码的聚类特征
        test_size (float): 测试集比例
        random_state (int): 随机种子
        output_dir (str): 输出目录
        filename_prefix (str): 输出文件名前缀，为None时自动生成
        threshold_zero (float): 砂厚小于此阈值视为0
        n_splits (int): 交叉验证折数
        model_params (dict): 可选的SVR参数，为None时使用GridSearchCV自动搜索
        verbose (bool): 是否输出详细信息

    返回:
        dict: 包含模型、标准化器、特征列表和评估指标的字典
    """
    print("======== SVR回归建模开始 ========")
    print(f"使用数据集大小: {len(data)} 行")

    # 复制数据，避免修改原始数据
    data_copy = data.copy()

    # 构建最终的特征列表
    final_features = selected_features.copy()

    # 如果使用One-Hot编码的聚类特征，添加到特征列表
    if use_onehot:
        cluster_features = [
            col for col in data_copy.columns if col.startswith("Cluster_")
        ]
        if cluster_features:
            if verbose:
                print(f"添加{len(cluster_features)}个One-Hot编码的聚类特征")
            final_features.extend(cluster_features)
        else:
            print("警告：数据中未找到One-Hot编码的聚类特征")

    # 准备特征和目标
    X = data_copy[final_features].values
    y = data_copy[target_column].values

    # 创建一个布尔变量，表示目标是否为0（或近似为0）
    y_is_zero = y <= threshold_zero

    # 如果需要，添加聚类概率作为附加特征
    if use_cluster_probs and "cluster_probs" in data_copy.columns:
        if verbose:
            print("添加聚类概率作为附加特征")

        # 假设cluster_probs是一个存储概率向量的列
        cluster_probs = np.array(data_copy["cluster_probs"].tolist())
        X = np.hstack((X, cluster_probs))

        # 更新特征列表（为了记录）
        final_features = final_features + [
            f"ClusterProb_{i}" for i in range(cluster_probs.shape[1])
        ]

    # 使用分层采样进行训练/测试集划分
    X_train, X_test, y_train, y_test, y_train_zero, y_test_zero = train_test_split(
        X,
        y,
        y_is_zero,
        test_size=test_size,
        random_state=random_state,
        stratify=y_is_zero,  # 使用是否为0作为分层依据
    )

    if verbose:
        print(f"训练集大小: {X_train.shape[0]} 样本")
        print(f"测试集大小: {X_test.shape[0]} 样本")
        print(f"训练集中砂厚为0的样本比例: {np.mean(y_train_zero):.2%}")
        print(f"测试集中砂厚为0的样本比例: {np.mean(y_test_zero):.2%}")
        print(f"使用特征数量: {X.shape[1]}")

    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 创建SVR模型
    if model_params is None:
        # 定义SVR参数网格
        param_grid = {
            "C": [0.1, 1, 10, 100],
            "epsilon": [0.01, 0.1, 0.2, 0.5],
            "kernel": ["linear", "rbf", "poly"],
            "gamma": ["scale", "auto", 0.1, 0.01],
        }

        svr = SVR()

        # 创建K折交叉验证
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # 使用GridSearchCV优化参数
        grid_search = GridSearchCV(
            svr,
            param_grid,
            cv=kfold,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=2 if verbose else 0,
        )

        # 训练模型
        print("\n正在训练SVR模型并优化参数...")
        grid_search.fit(X_train_scaled, y_train)

        # 获取最佳模型
        best_svr = grid_search.best_estimator_
        best_params = grid_search.best_params_

        if verbose:
            print("\n最佳SVR参数:")
            for param, value in best_params.items():
                print(f"  - {param}: {value}")
    else:
        # 使用指定的参数创建模型
        best_svr = SVR(**model_params)
        best_svr.fit(X_train_scaled, y_train)
        best_params = model_params

    # 在测试集上进行预测
    y_pred = best_svr.predict(X_test_scaled)

    # 计算评估指标
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n模型评估指标:")
    print(f"  - R²: {r2:.4f}")
    print(f"  - MAE: {mae:.4f}")
    print(f"  - RMSE: {rmse:.4f}")

    # 计算在"真实值=0"的样本中预测是否也为0的准确率
    y_pred_is_zero = y_pred <= threshold_zero
    zero_accuracy = accuracy_score(y_test_zero, y_pred_is_zero)

    # 计算更详细的指标
    # 真实为0，预测为0的比例 (真阴性率)
    true_negative_rate = np.sum(
        (y_test <= threshold_zero) & (y_pred <= threshold_zero)
    ) / np.sum(y_test <= threshold_zero)
    # 真实不为0，预测不为0的比例 (真阳性率)
    true_positive_rate = np.sum(
        (y_test > threshold_zero) & (y_pred > threshold_zero)
    ) / np.sum(y_test > threshold_zero)

    print(f"\n砂厚判别能力 (阈值 = {threshold_zero}):")
    print(f"  - 总体准确率: {zero_accuracy:.4f}")
    print(f"  - 真实为0时预测为0的比例: {true_negative_rate:.4f}")
    print(f"  - 真实不为0时预测不为0的比例: {true_positive_rate:.4f}")

    # 生成模型配置描述
    model_config = {
        "base_features": len(selected_features),
        "use_cluster_probs": use_cluster_probs,
        "use_onehot": use_onehot,
        "total_features": X.shape[1],
        "data_size": len(data),
    }

    # 模型保存文件名
    if filename_prefix is None:
        model_filename_base = f"SVR"
        if use_cluster_probs:
            model_filename_base += "_with_probs"
        if use_onehot:
            model_filename_base += "_with_onehot"
        model_filename_base += f"_{len(data)}_samples"
    else:
        model_filename_base = filename_prefix

    # 可视化预测结果
    visualize_model_results(
        y_test=y_test,
        y_pred=y_pred,
        y_test_zero=y_test_zero,
        y_pred_is_zero=y_pred_is_zero,
        threshold_zero=threshold_zero,
        output_dir=output_dir,
        filename_prefix=model_filename_base,
    )

    # 保存模型配置和评估指标
    model_results = {
        "model": best_svr,
        "scaler": scaler,
        "features": final_features,
        "config": model_config,
        "best_params": best_params,
        "metrics": {
            "r2": r2,
            "mae": mae,
            "rmse": rmse,
            "zero_accuracy": zero_accuracy,
            "true_negative_rate": true_negative_rate,
            "true_positive_rate": true_positive_rate,
        },
        "test_data": {
            "X_test": X_test,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_test_zero": y_test_zero,
            "y_pred_zero": y_pred_is_zero,
        },
    }

    print(f"======== SVR回归建模完成 ========")
    return model_results


def predict_with_model(
    model_results,
    data,
    coords_columns=["X", "Y", "Z"],
    output_dir="output",
    filename_prefix="predicted",
    threshold_zero=0.1,
    verbose=True,
):
    """
    使用训练好的模型对新数据进行预测

    参数:
        model_results (dict): build_regression_model返回的模型结果字典
        data (DataFrame): 包含特征的数据框
        coords_columns (list): 坐标列名称
        output_dir (str): 输出目录
        filename_prefix (str): 输出文件名前缀
        threshold_zero (float): 小于此值视为0
        verbose (bool): 是否输出详细信息

    返回:
        DataFrame: 包含预测结果的数据框
    """
    if verbose:
        print(f"======== 开始预测砂厚 ========")
        print(f"数据点数: {len(data)}")

    # 提取模型和标准化器
    model = model_results["model"]
    scaler = model_results["scaler"]
    features = model_results["features"]

    # 提取坐标信息
    coords_data = data[coords_columns].copy()

    # 准备特征数据
    # 检查特征是否全部存在于数据中
    missing_features = [
        f
        for f in features
        if f not in data.columns and not f.startswith("ClusterProb_")
    ]
    if missing_features:
        raise ValueError(f"数据中缺少以下特征: {missing_features}")

    # 预处理特征
    X_features = []

    # 处理基础特征和One-Hot编码特征
    base_features = [f for f in features if not f.startswith("ClusterProb_")]
    if base_features:
        X_base = data[base_features].values
        X_features.append(X_base)

    # 处理聚类概率特征（如果有）
    cluster_prob_features = [f for f in features if f.startswith("ClusterProb_")]
    if cluster_prob_features and "cluster_probs" in data.columns:
        if verbose:
            print("添加聚类概率作为预测特征")
        cluster_probs = np.array(data["cluster_probs"].tolist())
        X_features.append(cluster_probs)

    # 合并所有特征
    if len(X_features) > 1:
        X = np.hstack(X_features)
    else:
        X = X_features[0]

    # 标准化特征
    X_scaled = scaler.transform(X)

    # 使用模型进行预测
    if verbose:
        print(f"使用模型预测{len(X)}个点的砂厚...")

    predictions = model.predict(X_scaled)

    # 创建预测结果DataFrame
    prediction_results = pd.DataFrame(
        {
            **{col: coords_data[col] for col in coords_columns},
            "Predicted_Sand_Thickness": predictions,
        }
    )

    # 为负值预测结果置为0（如果有）
    if (prediction_results["Predicted_Sand_Thickness"] < 0).any():
        neg_count = (prediction_results["Predicted_Sand_Thickness"] < 0).sum()
        if verbose:
            print(f"有{neg_count}个点的预测砂厚为负值，已将其置为0")
        prediction_results["Predicted_Sand_Thickness"] = prediction_results[
            "Predicted_Sand_Thickness"
        ].clip(lower=0)

    # 统计预测结果
    pred_zero_count = (
        prediction_results["Predicted_Sand_Thickness"] <= threshold_zero
    ).sum()
    pred_nonzero_count = (
        prediction_results["Predicted_Sand_Thickness"] > threshold_zero
    ).sum()
    pred_mean = prediction_results["Predicted_Sand_Thickness"].mean()
    pred_max = prediction_results["Predicted_Sand_Thickness"].max()

    if verbose:
        print("\n预测结果统计:")
        print(f"  - 总点数: {len(prediction_results)}")
        print(
            f"  - 预测砂厚≤{threshold_zero}的点数: {pred_zero_count} ({pred_zero_count / len(prediction_results):.2%})"
        )
        print(
            f"  - 预测砂厚>{threshold_zero}的点数: {pred_nonzero_count} ({pred_nonzero_count / len(prediction_results):.2%})"
        )
        print(f"  - 平均预测砂厚: {pred_mean:.4f}")
        print(f"  - 最大预测砂厚: {pred_max:.4f}")

    # 保存预测结果到CSV文件
    prediction_file = os.path.join(output_dir, f"{filename_prefix}_sand_thickness.csv")
    prediction_results.to_csv(prediction_file, index=False)
    if verbose:
        print(f"\n预测结果已保存到文件: {prediction_file}")

    # 可视化预测结果 - 散点图
    visualize_predictions(
        prediction_results=prediction_results,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
        threshold_zero=threshold_zero,
    )

    # 导出为Petrel可读的XYZ格式文件
    export_to_petrel_format(
        prediction_results=prediction_results,
        coords_columns=coords_columns,
        output_dir=output_dir,
        filename_prefix=filename_prefix,
    )

    if verbose:
        print(f"======== 砂厚预测完成 ========")

    return prediction_results
