"""
PCA + Sigmoid 生成伪样本

工作流：
  1. 加载地震属性数据和井点数据
  2. 筛除离群井
  3. 地震属性预处理
  4. 根据井点分布缩小工区范围
  5. 提取井点处地震属性
  6. 筛选质量良好的属性（井-震统计对比）
  7. PCA 降维
  8. GMM 聚类
  9. Sigmoid 拟合（含智能虚拟点）
  10. 生成优化后的虚拟井

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
    p = argparse.ArgumentParser(description="PCA + Sigmoid 生成伪样本")
    p.add_argument("--seismic", required=True, help="地震属性 Surface 文件路径")
    p.add_argument("--wells", required=True, help="井点 xlsx 文件路径")
    p.add_argument("--surface", default="H6-2", help="目标层位名称")
    p.add_argument("--expansion-factor", type=float, default=1.5)
    p.add_argument("--pca-variance", type=float, default=0.9)
    p.add_argument("--n-clusters", type=int, default=2)
    p.add_argument("--sample-rows", type=int, default=40)
    p.add_argument("--sample-cols", type=int, default=40)
    p.add_argument("--max-samples-per-bin", type=int, default=30)
    return p.parse_args()


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
    # 步骤 9: Sigmoid 拟合
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 9: Sigmoid 拟合")
    print("=" * 60)

    sigmoid_data = pd.DataFrame()
    n_components = min(3, well_pca_features.shape[1])
    for i in range(n_components):
        sigmoid_data[f"PC{i + 1}"] = well_pca_features[:, i]
    sigmoid_data["Sand Thickness"] = data_well_purpose_surface_filtered["Sand Thickness"].values
    print(f"Sigmoid 建模数据形状: {sigmoid_data.shape}")

    pc_columns = [col for col in sigmoid_data.columns if col.startswith("PC")]
    sigmoid_model = SigmoidModel(
        data=sigmoid_data, feature_columns=pc_columns, target_column="Sand Thickness"
    )

    # 可视化原始样本分布
    visualize_feature_distribution(
        data=sigmoid_data, x_feature="PC1", y_feature="Sand Thickness",
        figsize=(10, 6), point_size=100, alpha=0.7, colormap="viridis",
        title="样本分布: PC1 vs Sand Thickness",
        save_path=os.path.join(figures_dir, "sigmoid_original_distribution.png"),
    )

    pc1_min, pc1_max = sigmoid_data["PC1"].min(), sigmoid_data["PC1"].max()
    pc1_median = sigmoid_data["PC1"].median()
    sand_thickness_max = sigmoid_data["Sand Thickness"].max()

    print(f"PC1 范围: {pc1_min:.2f} ~ {pc1_max:.2f}, 中位数: {pc1_median:.2f}")
    print(f"砂厚范围: {sigmoid_data['Sand Thickness'].min():.2f} ~ {sand_thickness_max:.2f} m")

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

    best_fit = fit_result
    best_model_name = "单特征(PC1)"

    if fit_result["success"]:
        print("\n=== 拟合成功! ===")
        sigmoid_model.visualize_fit(
            fit_result, figsize=(15, 6),
            save_path=os.path.join(figures_dir, "sigmoid_fit_result.png"),
        )
        params = fit_result["params"]
        param_errors = fit_result["param_errors"]
        for p in ["L", "k", "x0"]:
            print(f"  {p}: {params[p]:.4f} ± {param_errors[p + '_err']:.4f}")
        print(f"  R^2: {fit_result['r2_score']:.4f}")

        rmse = np.sqrt(np.mean((fit_result["y"] - fit_result["y_pred"]) ** 2))
        mae = np.mean(np.abs(fit_result["y"] - fit_result["y_pred"]))
        print(f"  RMSE: {rmse:.3f} m, MAE: {mae:.3f} m")

        # 保存模型摘要
        fit_summary = {
            "model_type": "sigmoid", "best_model": best_model_name,
            "features_used": str(best_fit["use_features"]),
            "feature_weights": str(best_fit.get("feature_weights", "None")),
            "n_samples": len(sigmoid_data),
            "n_virtual_points": len(sigmoid_model.current_data) - len(sigmoid_data),
            **best_fit["params"], **best_fit["param_errors"],
            "r2_score": best_fit["r2_score"], "rmse": rmse, "mae": mae,
        }
        pd.DataFrame([fit_summary]).to_csv(
            os.path.join(data_tmp_dir, "sigmoid_model_summary.csv"), index=False
        )

        # 全工区预测
        print("\n对全工区进行砂厚预测...")
        seismic_pca_features = pca_results["pca"].transform(pca_results["features_scaled"])
        seismic_pca_df = pd.DataFrame()
        for i in range(len(best_fit["use_features"])):
            seismic_pca_df[f"PC{i + 1}"] = seismic_pca_features[:, i]

        predicted_thickness = sigmoid_model.predict(
            seismic_pca_df, use_features=best_fit["use_features"],
            feature_weights=best_fit.get("feature_weights"),
        )

        prediction_results = pca_results["coords_clean"].copy()
        prediction_results["Predicted_Sand_Thickness"] = predicted_thickness
        prediction_results["Model_Type"] = best_model_name
        prediction_results["Model_R2"] = best_fit["r2_score"]
        prediction_results.to_csv(os.path.join(data_tmp_dir, "predicted_sand_thickness.csv"), index=False)

        print(f"  预测砂厚范围: {predicted_thickness.min():.2f} ~ {predicted_thickness.max():.2f} m")
        print(f"  预测砂厚均值: {predicted_thickness.mean():.2f} m")

        # 可视化预测结果
        visualize_attribute_map(
            data_points=prediction_results,
            attribute_name="Predicted_Sand_Thickness",
            attribute_label="预测砂厚 (m)",
            real_wells=data_well_purpose_surface_filtered,
            pseudo_wells=None, target_column="Sand Thickness",
            output_dir=figures_dir, filename_prefix="sigmoid_prediction",
            class_thresholds=CFG["class_thresholds"], figsize=(14, 10),
            dpi=150, cmap="viridis", point_size=50, well_size=100,
        )

        # 井点预测 vs 真实值
        well_pca_df = pd.DataFrame()
        for i in range(len(best_fit["use_features"])):
            well_pca_df[f"PC{i + 1}"] = well_pca_features[:, i]
        well_predictions = sigmoid_model.predict(
            well_pca_df, use_features=best_fit["use_features"],
            feature_weights=best_fit.get("feature_weights"),
        )
        sigmoid_model.visualize_predict(
            true_values=sigmoid_data["Sand Thickness"].values,
            predicted_values=well_predictions,
            feature_values=well_pca_features[:, 0],
            show_confidence_band=True,
            save_path=os.path.join(figures_dir, "sigmoid_prediction_detailed_analysis.png"),
        )
    else:
        print(f"\n=== 拟合失败: {fit_result.get('error', 'Unknown')} ===")
        return

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
    seismic_samples.to_csv(os.path.join(data_tmp_dir, "seismic_samples.csv"), index=False)

    print(f"采样点数: {len(seismic_samples)}")

    # Sigmoid 预测虚拟井砂厚
    sample_features = seismic_samples[pca_results["features_clean"].columns].values
    sample_features_scaled = pca_results["scaler"].transform(sample_features)
    sample_pca_features = pca_results["pca"].transform(sample_features_scaled)

    sample_pca_df = pd.DataFrame()
    for i in range(len(best_fit["use_features"])):
        sample_pca_df[f"PC{i + 1}"] = sample_pca_features[:, i]

    predicted_sample_thickness = sigmoid_model.predict(
        sample_pca_df, use_features=best_fit["use_features"],
        feature_weights=best_fit.get("feature_weights"),
    )
    seismic_samples["Predicted_Sand_Thickness"] = predicted_sample_thickness
    neg_count = (predicted_sample_thickness < 0).sum()
    if neg_count > 0:
        seismic_samples["Predicted_Sand_Thickness"] = seismic_samples["Predicted_Sand_Thickness"].clip(lower=0)
        print(f"注意: {neg_count} 个负值预测已替换为 0")

    print(f"虚拟井砂厚: {seismic_samples['Predicted_Sand_Thickness'].min():.2f} ~ "
          f"{seismic_samples['Predicted_Sand_Thickness'].max():.2f} m, "
          f"均值: {seismic_samples['Predicted_Sand_Thickness'].mean():.2f} m")

    # ================================================================
    # 步骤 11: 虚拟井三层优化筛选
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 11: 虚拟井优化筛选")
    print("=" * 60)

    pseudo_wells_data = seismic_samples.copy()
    real_wells_data = data_well_purpose_surface_filtered.copy()
    print(f"初始虚拟井: {len(pseudo_wells_data)}")

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
        final_indices.extend(chosen)
        print(f"  区间 {bin_labels[i]}: {len(bin_indices)} -> {len(chosen)}")

    optimized_pseudo_wells = layer2.loc[final_indices].copy().reset_index(drop=True)
    print(f"\n虚拟井优化结果: {len(pseudo_wells_data)} -> {len(layer1)} -> {len(layer2)} -> {len(optimized_pseudo_wells)}")

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

    seismic_features_all = data_seismic_attr_filtered[pca_results["features_clean"].columns].values
    seismic_features_all_scaled = pca_results["scaler"].transform(seismic_features_all)
    seismic_pca_all = pca_results["pca"].transform(seismic_features_all_scaled)

    seismic_pca_all_df = pd.DataFrame()
    for i in range(len(best_fit["use_features"])):
        seismic_pca_all_df[f"PC{i + 1}"] = seismic_pca_all[:, i]

    seismic_pred = sigmoid_model.predict(
        seismic_pca_all_df, use_features=best_fit["use_features"],
        feature_weights=best_fit.get("feature_weights"),
    )
    seismic_pred = np.maximum(seismic_pred, 0)

    data_with_pred = data_seismic_attr_filtered.copy()
    data_with_pred["Predicted_Sand_Thickness"] = seismic_pred

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
