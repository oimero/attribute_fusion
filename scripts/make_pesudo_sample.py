"""
PCA + 三模型共识生成伪样本

工作流：
  1. 加载地震属性数据和井点数据
  2. 筛除离群井
  3. 地震属性预处理
  4. 根据井点分布缩小工区范围
  5. 提取井点处地震属性
  6. 筛选质量良好的属性（井-震统计对比）
  7. PCA 降维
  8. GMM 聚类
  9. Ridge、Lasso、Sigmoid 拟合
  10. 三模型共识筛选并生成优化后的虚拟井

用法：
  python scripts/make_pesudo_sample.py \
      --seismic data/target/H6-2 \
      --wells scripts/output/well_data_preprocess_20260701_160047/well_horizon_processed.xlsx \
      --surface H6-2
"""

import argparse
import os
import sys
import warnings
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# 将项目根目录加入路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# ================================================================
# 配置参数
# ================================================================
CFG = {
    # -- 地震属性预处理
    "preprocess": {
        "missing_values": [-999],
        "missing_threshold": 0.6,
        "outlier_method": "iqr",
        "outlier_threshold": 2.0,
        "outlier_treatment": "clip",
    },
    # -- 井点地震属性提取
    "well_attr_extract": {
        "max_distance": 50,
        "num_points": 10,
    },
    # -- 井-震属性质量筛选
    "attr_quality": {
        "ratio_threshold": 5.0,
        "range_ratio_threshold": 10.0,
        "std_ratio_threshold": 10.0,
    },
    # -- Sigmoid 虚拟点配置
    "virtual_points": {
        "placement_strategy": "conservative",
        "n_points": 10,
        "noise_factor": 0.05,
        "auto_detect": True,
    },
    # -- Sigmoid 拟合边界/初值 (相对于数据范围的系数)
    "sigmoid_fit": {
        "use_features": ["PC1"],
        "bounds_L_min_factor": 0.2,     # L 下界 = max_sand * factor
        "bounds_L_max_factor": 3.0,     # L 上界 = max_sand * factor
        "bounds_k_min": -10,
        "bounds_k_max": 10,
        "bounds_x0_margin_factor": 1.0,  # x0 边界 = data_range * factor
        "initial_L_factor": 0.7,          # 初值 L = max_sand * factor
        "initial_k": 1.0,
        "max_iterations": 3000,
    },
    # -- Ridge / Lasso / Sigmoid 共识配置
    "multi_model_fit": {
        "ridge_alpha": 1.0,
        "lasso_alpha": 0.1,
        "lasso_max_iterations": 5000,
        "random_seed": 42,
        "max_prediction_spread": 5.0,
    },
    # -- 等间距采样网格
    "sample_grid": {
        "n_rows": 40,
        "n_cols": 40,
    },
    # -- 虚拟井优化筛选
    "pseudo_optimize": {
        "proximity_radius": 200,        # 第一层：距真实井距离阈值 (m)
        "max_thickness_diff": 5.0,      # 第一层：砂厚差异阈值 (m)
        "min_pseudo_distance": 200,     # 第二层：虚拟井间最小距离 (m)
        "thickness_bins": [0, 1, 13.75, np.inf],  # 第三层：砂厚区间
        "max_samples_per_bin": 30,      # 第三层：每区间最大样本数
    },
    # -- 可视化
    "class_thresholds": [1, 13.75],
}
# ================================================================

from src.data_utils import (
    extract_seismic_attributes_for_wells,
    extract_uniform_seismic_samples,
    filter_anomalous_attributes,
    filter_outlier_wells,
    filter_seismic_by_wells,
    identify_attributes,
    parse_petrel_file,
    preprocess_features,
)
from src.gmm_clustering import perform_gmm_clustering, visualize_gmm_clustering
from src.pca_analysis import perform_pca_analysis, visualize_pca_clustering
from src.sigmoid import SigmoidModel
from src.visualization import visualize_attribute_map, visualize_feature_distribution

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False


def setup_output_dir(script_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(PROJECT_ROOT, "scripts", "output")
    output_dir = os.path.join(output_root, f"{script_name}_{timestamp}")
    figures_dir = os.path.join(output_dir, "figures")
    data_tmp_dir = os.path.join(output_dir, "data_tmp")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(data_tmp_dir, exist_ok=True)
    return output_dir, figures_dir, data_tmp_dir


def parse_args():
    p = argparse.ArgumentParser(description="PCA + 三模型共识生成伪样本")
    p.add_argument("--seismic", required=True, help="地震属性 Surface 文件路径")
    p.add_argument("--wells", required=True, help="井点 xlsx 文件路径")
    p.add_argument("--surface", required=True, help="目标层位名称")
    p.add_argument("--expansion-factor", type=float, default=1.5)
    p.add_argument("--pca-variance", type=float, default=0.9)
    p.add_argument("--n-clusters", type=int, default=2)
    p.add_argument("--sample-rows", type=int, default=40)
    p.add_argument("--sample-cols", type=int, default=40)
    p.add_argument("--max-samples-per-bin", type=int, default=30)
    return p.parse_args()


MODEL_PREDICTION_COLUMNS = [
    "Ridge_Prediction",
    "Lasso_Prediction",
    "Sigmoid_Prediction",
]


def calculate_model_consensus(
    ridge_predictions,
    lasso_predictions,
    sigmoid_predictions,
    max_prediction_spread,
):
    """截断三模型负预测，并计算逐点均值、极差和共识标记。"""
    prediction_matrix = np.column_stack(
        [ridge_predictions, lasso_predictions, sigmoid_predictions]
    ).astype(float)
    if prediction_matrix.ndim != 2 or prediction_matrix.shape[1] != 3:
        raise ValueError("三模型预测必须能组成 n×3 数组")
    if not np.isfinite(prediction_matrix).all():
        raise ValueError("三模型预测包含 NaN 或无穷值，无法生成共识伪样本")

    prediction_matrix = np.maximum(prediction_matrix, 0.0)
    result = pd.DataFrame(prediction_matrix, columns=MODEL_PREDICTION_COLUMNS)
    result["Prediction_Spread"] = prediction_matrix.max(axis=1) - prediction_matrix.min(axis=1)
    result["Predicted_Sand_Thickness"] = prediction_matrix.mean(axis=1)
    result["Model_Agreement"] = result["Prediction_Spread"] <= max_prediction_spread
    return result


def predict_with_three_models(
    pca_features,
    pc_columns,
    ridge_model,
    lasso_model,
    sigmoid_model,
    sigmoid_fit,
    max_prediction_spread,
):
    """使用三个已拟合模型预测，并返回统一的共识结果。"""
    pca_features = np.asarray(pca_features)
    if pca_features.ndim != 2 or pca_features.shape[1] < len(pc_columns):
        raise ValueError("PCA 特征维度不足，无法进行三模型预测")

    linear_features = pca_features[:, : len(pc_columns)]
    sigmoid_features = pd.DataFrame(linear_features, columns=pc_columns)
    ridge_predictions = ridge_model.predict(linear_features)
    lasso_predictions = lasso_model.predict(linear_features)
    sigmoid_predictions = sigmoid_model.predict(
        sigmoid_features,
        use_features=sigmoid_fit["use_features"],
        feature_weights=sigmoid_fit.get("feature_weights"),
    )
    return calculate_model_consensus(
        ridge_predictions,
        lasso_predictions,
        sigmoid_predictions,
        max_prediction_spread,
    )


def calculate_regression_metrics(true_values, predicted_values):
    """计算统一口径的真实井拟合指标。"""
    return {
        "r2_score": r2_score(true_values, predicted_values),
        "rmse": np.sqrt(mean_squared_error(true_values, predicted_values)),
        "mae": mean_absolute_error(true_values, predicted_values),
    }


def main():
    args = parse_args()
    SURFACE_NAME = args.surface
    output_dir, figures_dir, data_tmp_dir = setup_output_dir("make_pesudo_sample")

    print(f"输出目录: {output_dir}")
    print(f"图件目录: {figures_dir}")
    print(f"临时数据: {data_tmp_dir}")

    # ================================================================
    # 步骤 1: 导入数据
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 1: 导入数据")
    print("=" * 60)

    print(f"地震数据路径: {args.seismic}")
    data_seismic_attr = parse_petrel_file(args.seismic)

    data_well_position = pd.read_excel(args.wells)
    data_well_purpose_surface_position = (
        data_well_position[data_well_position["Surface"] == SURFACE_NAME]
        .replace(-999, np.nan)
        .dropna(subset=["Sand Thickness"])
        .reset_index(drop=True)
    )
    print(f"层位 {SURFACE_NAME} 的井点数量: {len(data_well_purpose_surface_position)}")

    # ================================================================
    # 步骤 2: 筛除离群井
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 2: 筛除离群井")
    print("=" * 60)

    data_well_purpose_surface_filtered = filter_outlier_wells(
        data_well_purpose_surface_position, method="iqr"
    )
    print(f"筛选前井点数量: {len(data_well_purpose_surface_position)}")
    print(f"筛选后井点数量: {len(data_well_purpose_surface_filtered)}")

    # 可视化
    x_min, x_max = data_well_purpose_surface_position["X"].min(), data_well_purpose_surface_position["X"].max()
    y_min, y_max = data_well_purpose_surface_position["Y"].min(), data_well_purpose_surface_position["Y"].max()
    margin = 0.05
    x_range, y_range = x_max - x_min, y_max - y_min
    x_min -= x_range * margin; x_max += x_range * margin
    y_min -= y_range * margin; y_max += y_range * margin

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.scatter(data_well_purpose_surface_position["X"], data_well_purpose_surface_position["Y"], c="blue")
    ax1.set_title("筛选前井点分布"); ax1.set_xlabel("X坐标"); ax1.set_ylabel("Y坐标")
    ax1.set_xlim(x_min, x_max); ax1.set_ylim(y_min, y_max)
    ax2.scatter(data_well_purpose_surface_filtered["X"], data_well_purpose_surface_filtered["Y"], c="red")
    ax2.set_title("筛选后井点分布"); ax2.set_xlabel("X坐标"); ax2.set_ylabel("Y坐标")
    ax2.set_xlim(x_min, x_max); ax2.set_ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "well_filtering_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  图件: well_filtering_comparison.png")

    # ================================================================
    # 步骤 3: 处理地震属性缺失值 / 离群值
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 3: 地震属性预处理")
    print("=" * 60)

    attribute_names, _ = identify_attributes(args.seismic)
    p = CFG["preprocess"]
    processed_features, stats, report = preprocess_features(
        data=data_seismic_attr, attribute_columns=attribute_names,
        missing_values=p["missing_values"], missing_threshold=p["missing_threshold"],
        outlier_method=p["outlier_method"], outlier_threshold=p["outlier_threshold"],
        outlier_treatment=p["outlier_treatment"], verbose=True,
    )
    attribute_names_processed = list(processed_features.columns)
    data_seismic_attr_processed = data_seismic_attr[["X", "Y"]].copy()
    for col in processed_features.columns:
        data_seismic_attr_processed[col] = processed_features[col]

    # ================================================================
    # 步骤 4: 限制工区范围
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 4: 根据井点分布缩小工区范围")
    print("=" * 60)

    data_seismic_attr_filtered, area_bounds = filter_seismic_by_wells(
        seismic_data=data_seismic_attr_processed,
        well_data=data_well_purpose_surface_filtered,
        expansion_factor=args.expansion_factor,
        plot=True, output_dir=figures_dir,
    )

    # ================================================================
    # 步骤 5: 提取井点处地震属性
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 5: 提取井点处地震属性")
    print("=" * 60)

    w = CFG["well_attr_extract"]
    data_well_attr_filtered = extract_seismic_attributes_for_wells(
        well_data=data_well_purpose_surface_filtered,
        seismic_data=data_seismic_attr_processed,
        max_distance=w["max_distance"], num_points=w["num_points"],
    )
    well_attr_path = os.path.join(data_tmp_dir, f"{SURFACE_NAME.replace('-', '_')}_wells_attr_filtered.xlsx")
    data_well_attr_filtered.to_excel(well_attr_path, index=False)
    print(f"井点属性已保存: {well_attr_path}")

    # ================================================================
    # 步骤 6: 筛选质量良好的属性
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 6: 井-震统计对比，筛选质量良好的属性")
    print("=" * 60)

    q = CFG["attr_quality"]
    good_attributes, anomalous_attributes, attribute_stats = filter_anomalous_attributes(
        seismic_data=data_seismic_attr_filtered,
        well_data=data_well_attr_filtered,
        common_attributes=attribute_names_processed,
        ratio_threshold=q["ratio_threshold"],
        range_ratio_threshold=q["range_ratio_threshold"],
        std_ratio_threshold=q["std_ratio_threshold"],
        verbose=True,
    )
    print(f"\n保留 {len(good_attributes)} 个质量良好的属性")

    # ================================================================
    # 步骤 7: PCA 降维
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 7: PCA 降维")
    print("=" * 60)

    pca_results = perform_pca_analysis(
        data=data_seismic_attr_filtered,
        attribute_columns=good_attributes,
        variance_threshold=args.pca_variance,
        output_dir=figures_dir,
    )

    # ================================================================
    # 步骤 8: GMM 聚类
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 8: GMM 聚类")
    print("=" * 60)

    best_n = args.n_clusters
    gmm_results = perform_gmm_clustering(
        features=pca_results["features_pca"],
        coords=pca_results["coords_clean"],
        n_clusters=best_n,
    )
    gmm_results["result_df"].to_csv(os.path.join(data_tmp_dir, "gmm_best_clusters.csv"), index=False)

    # 井点投影到 PCA 空间
    well_features = data_well_attr_filtered[pca_results["features_clean"].columns].values
    well_features_scaled = pca_results["scaler"].transform(well_features)
    well_pca_features = pca_results["pca"].transform(well_features_scaled)

    visualize_pca_clustering(
        features_pca=pca_results["features_pca"],
        cluster_labels=gmm_results["cluster_labels"],
        n_clusters=best_n, output_dir=figures_dir,
        well_data=data_well_purpose_surface_filtered,
        well_pca_features=well_pca_features,
        target_column="Sand Thickness", class_thresholds=CFG["class_thresholds"],
    )

    visualize_gmm_clustering(
        clustering_results=gmm_results, output_dir=figures_dir,
        prefix="pca", well_data=data_well_purpose_surface_filtered,
        target_column="Sand Thickness", class_thresholds=CFG["class_thresholds"],
        point_size=10, well_size=50,
    )

    # ================================================================
    # 步骤 9: 三模型拟合
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 9: Ridge、Lasso、Sigmoid 拟合")
    print("=" * 60)

    modeling_data = pd.DataFrame()
    n_components = min(3, well_pca_features.shape[1])
    for i in range(n_components):
        modeling_data[f"PC{i + 1}"] = well_pca_features[:, i]
    modeling_data["Sand Thickness"] = data_well_purpose_surface_filtered["Sand Thickness"].values
    print(f"三模型建模数据形状: {modeling_data.shape}")

    pc_columns = [col for col in modeling_data.columns if col.startswith("PC")]
    target_values = modeling_data["Sand Thickness"].values
    linear_features = modeling_data[pc_columns].values
    mf = CFG["multi_model_fit"]

    try:
        ridge_model = Ridge(
            alpha=mf["ridge_alpha"], random_state=mf["random_seed"]
        ).fit(linear_features, target_values)
        lasso_model = Lasso(
            alpha=mf["lasso_alpha"],
            random_state=mf["random_seed"],
            max_iter=mf["lasso_max_iterations"],
        ).fit(linear_features, target_values)
    except Exception as exc:
        raise RuntimeError(f"Ridge/Lasso 拟合失败: {exc}") from exc

    sigmoid_model = SigmoidModel(
        data=modeling_data, feature_columns=pc_columns, target_column="Sand Thickness"
    )

    # 可视化原始样本分布
    visualize_feature_distribution(
        data=modeling_data, x_feature="PC1", y_feature="Sand Thickness",
        figsize=(10, 6), point_size=100, alpha=0.7, colormap="viridis",
        title="样本分布: PC1 vs Sand Thickness",
        save_path=os.path.join(figures_dir, "sigmoid_original_distribution.png"),
    )

    pc1_min, pc1_max = modeling_data["PC1"].min(), modeling_data["PC1"].max()
    pc1_median = modeling_data["PC1"].median()
    sand_thickness_max = modeling_data["Sand Thickness"].max()

    print(f"PC1 范围: {pc1_min:.2f} ~ {pc1_max:.2f}, 中位数: {pc1_median:.2f}")
    print(f"砂厚范围: {modeling_data['Sand Thickness'].min():.2f} ~ {sand_thickness_max:.2f} m")

    virtual_config = CFG["virtual_points"]

    sf = CFG["sigmoid_fit"]
    bounds = (
        [sand_thickness_max * sf["bounds_L_min_factor"],
         sf["bounds_k_min"],
         pc1_min - (pc1_max - pc1_min) * sf["bounds_x0_margin_factor"]],
        [sand_thickness_max * sf["bounds_L_max_factor"],
         sf["bounds_k_max"],
         pc1_max + (pc1_max - pc1_min) * sf["bounds_x0_margin_factor"]],
    )
    initial_guess = [sand_thickness_max * sf["initial_L_factor"], sf["initial_k"], pc1_median]

    fit_result = sigmoid_model.fit(
        use_features=sf["use_features"],
        virtual_points_config=virtual_config,
        bounds=bounds,
        initial_guess=initial_guess,
        max_iterations=sf["max_iterations"],
    )

    if not fit_result["success"]:
        raise RuntimeError(f"Sigmoid 拟合失败: {fit_result.get('error', 'Unknown')}")

    best_fit = fit_result
    print("\n=== 三模型拟合成功! ===")
    sigmoid_model.visualize_fit(
        fit_result, figsize=(15, 6),
        save_path=os.path.join(figures_dir, "sigmoid_fit_result.png"),
    )
    params = fit_result["params"]
    param_errors = fit_result["param_errors"]
    for p in ["L", "k", "x0"]:
        print(f"  Sigmoid {p}: {params[p]:.4f} ± {param_errors[p + '_err']:.4f}")

    # 真实井统一口径预测与拟合指标
    well_predictions = predict_with_three_models(
        well_pca_features[:, :n_components], pc_columns,
        ridge_model, lasso_model, sigmoid_model, best_fit,
        mf["max_prediction_spread"],
    )
    model_metrics = {}
    for prediction_column in MODEL_PREDICTION_COLUMNS:
        model_name = prediction_column.replace("_Prediction", "")
        model_metrics[model_name] = calculate_regression_metrics(
            target_values, well_predictions[prediction_column].values
        )
        metrics = model_metrics[model_name]
        print(
            f"  {model_name}: R^2={metrics['r2_score']:.4f}, "
            f"RMSE={metrics['rmse']:.3f} m, MAE={metrics['mae']:.3f} m"
        )

    sigmoid_model.visualize_predict(
        true_values=target_values,
        predicted_values=well_predictions["Sigmoid_Prediction"].values,
        feature_values=well_pca_features[:, 0],
        show_confidence_band=True,
        save_path=os.path.join(figures_dir, "sigmoid_prediction_detailed_analysis.png"),
    )

    # 全工区三模型预测与共识诊断
    print("\n对全工区进行三模型砂厚预测...")
    seismic_pca_features = pca_results["pca"].transform(pca_results["features_scaled"])
    full_consensus = predict_with_three_models(
        seismic_pca_features[:, :n_components], pc_columns,
        ridge_model, lasso_model, sigmoid_model, best_fit,
        mf["max_prediction_spread"],
    )
    prediction_results = pd.concat(
        [
            pca_results["coords_clean"].reset_index(drop=True),
            full_consensus.reset_index(drop=True),
        ],
        axis=1,
    )
    prediction_results.to_csv(
        os.path.join(data_tmp_dir, "predicted_sand_thickness.csv"), index=False
    )

    full_agreement_rate = full_consensus["Model_Agreement"].mean()
    consensus_values = full_consensus["Predicted_Sand_Thickness"]
    print(
        f"  三模型均值范围: {consensus_values.min():.2f} ~ "
        f"{consensus_values.max():.2f} m, 均值: {consensus_values.mean():.2f} m"
    )
    print(
        f"  共识通过率（最大差 ≤ {mf['max_prediction_spread']:.1f}m）: "
        f"{full_agreement_rate:.1%}"
    )

    summary_rows = [
        {
            "model": "Ridge",
            "features_used": str(pc_columns),
            "parameters": str({
                "alpha": mf["ridge_alpha"],
                "intercept": ridge_model.intercept_,
                "coefficients": ridge_model.coef_.tolist(),
            }),
            "n_real_samples": len(modeling_data),
            "n_virtual_points": 0,
            **model_metrics["Ridge"],
            "full_area_agreement_rate": full_agreement_rate,
        },
        {
            "model": "Lasso",
            "features_used": str(pc_columns),
            "parameters": str({
                "alpha": mf["lasso_alpha"],
                "intercept": lasso_model.intercept_,
                "coefficients": lasso_model.coef_.tolist(),
            }),
            "n_real_samples": len(modeling_data),
            "n_virtual_points": 0,
            **model_metrics["Lasso"],
            "full_area_agreement_rate": full_agreement_rate,
        },
        {
            "model": "Sigmoid",
            "features_used": str(best_fit["use_features"]),
            "parameters": str(best_fit["params"]),
            "n_real_samples": len(modeling_data),
            "n_virtual_points": len(sigmoid_model.current_data) - len(modeling_data),
            **model_metrics["Sigmoid"],
            "full_area_agreement_rate": full_agreement_rate,
        },
    ]
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(data_tmp_dir, "model_fit_summary.csv"), index=False
    )

    visualize_attribute_map(
        data_points=prediction_results,
        attribute_name="Predicted_Sand_Thickness",
        attribute_label="三模型平均预测砂厚 (m)",
        real_wells=data_well_purpose_surface_filtered,
        pseudo_wells=None, target_column="Sand Thickness",
        output_dir=figures_dir, filename_prefix="multi_model_mean_prediction",
        class_thresholds=CFG["class_thresholds"], figsize=(14, 10),
        dpi=150, cmap="viridis", point_size=50, well_size=100,
    )
    visualize_attribute_map(
        data_points=prediction_results,
        attribute_name="Prediction_Spread",
        attribute_label="三模型预测最大差 (m)",
        real_wells=data_well_purpose_surface_filtered,
        pseudo_wells=None, target_column="Sand Thickness",
        output_dir=figures_dir, filename_prefix="multi_model_prediction_spread",
        class_thresholds=CFG["class_thresholds"], figsize=(14, 10),
        dpi=150, cmap="magma", point_size=50, well_size=100,
    )

    # ================================================================
    # 步骤 10: 生成虚拟井
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 10: 生成虚拟井")
    print("=" * 60)

    # 等间距采样
    seismic_samples = extract_uniform_seismic_samples(
        seismic_data=data_seismic_attr_filtered,
        n_rows=args.sample_rows, n_cols=args.sample_cols,
        area_bounds=area_bounds,
    )

    # 可视化采样点
    fig, ax = plt.subplots(figsize=(15, 10))
    sample_ratio_plt = min(1.0, 5000 / len(data_seismic_attr_filtered))
    seismic_sample_plt = data_seismic_attr_filtered.sample(frac=sample_ratio_plt)
    ax.scatter(seismic_sample_plt["X"], seismic_sample_plt["Y"],
               color="lightgray", alpha=0.3, s=10, label="地震数据(抽样)")
    ax.scatter(data_well_purpose_surface_filtered["X"], data_well_purpose_surface_filtered["Y"],
               color="red", s=100, marker="^", label="真实井点")
    ax.scatter(seismic_samples["X"], seismic_samples["Y"],
               color="blue", s=50, marker="o", label="等间距采样点")
    ax.set_title("真实井点与等间距采样点分布"); ax.set_xlabel("X坐标"); ax.set_ylabel("Y坐标")
    ax.legend(loc="upper right"); ax.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(figures_dir, "real_wells_and_seismic_samples.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"采样点数: {len(seismic_samples)}")

    # 三模型预测虚拟井砂厚
    sample_features = seismic_samples[pca_results["features_clean"].columns].values
    sample_features_scaled = pca_results["scaler"].transform(sample_features)
    sample_pca_features = pca_results["pca"].transform(sample_features_scaled)

    sample_consensus = predict_with_three_models(
        sample_pca_features[:, :n_components], pc_columns,
        ridge_model, lasso_model, sigmoid_model, best_fit,
        mf["max_prediction_spread"],
    )
    seismic_samples = pd.concat(
        [seismic_samples.reset_index(drop=True), sample_consensus.reset_index(drop=True)],
        axis=1,
    )
    seismic_samples.to_csv(
        os.path.join(data_tmp_dir, "seismic_samples.csv"), index=False
    )

    print(f"三模型共识均值: {seismic_samples['Predicted_Sand_Thickness'].min():.2f} ~ "
          f"{seismic_samples['Predicted_Sand_Thickness'].max():.2f} m, "
          f"均值: {seismic_samples['Predicted_Sand_Thickness'].mean():.2f} m")

    # ================================================================
    # 步骤 11: 虚拟井三层优化筛选
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 11: 虚拟井优化筛选")
    print("=" * 60)

    initial_pseudo_count = len(seismic_samples)
    agreement_mask = seismic_samples["Model_Agreement"]
    pseudo_wells_data = seismic_samples[agreement_mask].copy().reset_index(drop=True)
    real_wells_data = data_well_purpose_surface_filtered.copy()
    rejected_by_agreement = initial_pseudo_count - len(pseudo_wells_data)
    print(f"初始候选井: {initial_pseudo_count}")
    print(
        f"共识筛选（最大差 ≤ {mf['max_prediction_spread']:.1f}m）: "
        f"排除 {rejected_by_agreement} 个点, 剩余 {len(pseudo_wells_data)}"
    )
    if pseudo_wells_data.empty:
        raise RuntimeError(
            "三模型共识筛选后没有候选井；不会自动放宽 max_prediction_spread"
        )

    # 第一层：排除靠近真实井且砂厚差异大的点
    po = CFG["pseudo_optimize"]
    proximity_radius = po["proximity_radius"]
    max_thickness_diff = po["max_thickness_diff"]
    real_coords = real_wells_data[["X", "Y"]].values
    real_thickness = real_wells_data["Sand Thickness"].values
    pseudo_coords = pseudo_wells_data[["X", "Y"]].values
    pseudo_thickness = pseudo_wells_data["Predicted_Sand_Thickness"].values

    distances = cdist(pseudo_coords, real_coords)
    min_distances = np.min(distances, axis=1)
    closest_well_indices = np.argmin(distances, axis=1)

    exclude_mask = np.zeros(len(pseudo_wells_data), dtype=bool)
    for i in range(len(pseudo_wells_data)):
        if min_distances[i] <= proximity_radius:
            if abs(pseudo_thickness[i] - real_thickness[closest_well_indices[i]]) > max_thickness_diff:
                exclude_mask[i] = True
    layer1 = pseudo_wells_data[~exclude_mask].copy().reset_index(drop=True)
    print(f"第一层: 排除 {(exclude_mask).sum()} 个点, 剩余 {len(layer1)}")
    if layer1.empty:
        raise RuntimeError("近井差异筛选后没有候选井，无法继续生成虚拟井")

    # 第二层：贪心距离选择
    min_pseudo_distance = po["min_pseudo_distance"]
    layer1_coords = layer1[["X", "Y"]].values
    pseudo_distances = cdist(layer1_coords, layer1_coords)
    thickness_order = np.argsort(layer1["Predicted_Sand_Thickness"].values)

    selected = []
    for idx in thickness_order:
        if not any(pseudo_distances[idx, s] < min_pseudo_distance for s in selected):
            selected.append(idx)
    layer2 = layer1.iloc[selected].copy().reset_index(drop=True)
    print(f"第二层: 选择 {len(layer2)} 个点")

    # 第三层：砂厚分布均衡
    thickness_bins = po["thickness_bins"]
    bin_labels = ["0-1m", "1-13.75m", ">13.75m"]
    max_samples_per_bin = args.max_samples_per_bin or po["max_samples_per_bin"]

    final_indices = []
    for i in range(len(thickness_bins) - 1):
        bin_mask = (layer2["Predicted_Sand_Thickness"] >= thickness_bins[i]) & (
            layer2["Predicted_Sand_Thickness"] < thickness_bins[i + 1]
        )
        bin_indices = layer2.index[bin_mask].tolist()
        if len(bin_indices) > max_samples_per_bin:
            chosen = np.random.choice(bin_indices, max_samples_per_bin, replace=False).tolist()
        else:
            chosen = bin_indices
            if len(bin_indices) < max_samples_per_bin:
                print(
                    f"  警告: 区间 {bin_labels[i]} 仅有 {len(bin_indices)} 个候选点，"
                    "保持三模型共识门槛，不自动放宽"
                )
        final_indices.extend(chosen)
        print(f"  区间 {bin_labels[i]}: {len(bin_indices)} -> {len(chosen)}")

    optimized_pseudo_wells = layer2.loc[final_indices].copy().reset_index(drop=True)
    print(
        f"\n虚拟井优化结果: {initial_pseudo_count} -> 共识 {len(pseudo_wells_data)} "
        f"-> 近井 {len(layer1)} -> 空间 {len(layer2)} -> 最终 {len(optimized_pseudo_wells)}"
    )
    if optimized_pseudo_wells.empty:
        raise RuntimeError("优化筛选后没有虚拟井，未生成空结果文件")

    if not (
        optimized_pseudo_wells["Prediction_Spread"] <= mf["max_prediction_spread"]
    ).all():
        raise RuntimeError("内部错误: 最终虚拟井中存在未通过三模型共识门槛的样本")

    # 保存最终虚拟井
    pseudo_output_path = os.path.join(output_dir, f"{SURFACE_NAME.replace('-', '_')}_optimized_pseudo_wells.csv")
    optimized_pseudo_wells.to_csv(pseudo_output_path, index=False)
    print(f"优化虚拟井已保存: {pseudo_output_path}")

    for i in range(len(thickness_bins) - 1):
        cnt = ((optimized_pseudo_wells["Predicted_Sand_Thickness"] >= thickness_bins[i]) &
               (optimized_pseudo_wells["Predicted_Sand_Thickness"] < thickness_bins[i + 1])).sum()
        pct = cnt / len(optimized_pseudo_wells) * 100 if len(optimized_pseudo_wells) > 0 else 0
        print(f"  {bin_labels[i]}: {cnt} 个 ({pct:.1f}%)")

    # ================================================================
    # 步骤 12: 最终可视化
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 12: 最终可视化")
    print("=" * 60)

    data_with_pred = data_seismic_attr_filtered.copy()
    for column in MODEL_PREDICTION_COLUMNS + [
        "Prediction_Spread", "Predicted_Sand_Thickness", "Model_Agreement"
    ]:
        data_with_pred[column] = prediction_results[column].values

    visualize_attribute_map(
        data_points=data_with_pred,
        attribute_name="Predicted_Sand_Thickness",
        attribute_label="预测砂厚 (m)",
        real_wells=data_well_purpose_surface_filtered,
        pseudo_wells=optimized_pseudo_wells,
        target_column="Sand Thickness", output_dir=figures_dir,
        filename_prefix="pseudo_wells_optimized",
        class_thresholds=CFG["class_thresholds"], figsize=(10, 8),
        dpi=150, cmap="viridis", point_size=50, well_size=100,
    )

    print("\n" + "=" * 60)
    print(f"全部完成！输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
