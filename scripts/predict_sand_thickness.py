"""
伪样本 + SVR 集成预测砂厚

工作流：
  1. 加载地震属性数据、真实井点、虚拟井点
  2. 地震属性预处理
  3. 特征自相关性分组
  4. 动态数据增强（按砂厚区间均衡）
  5. 遍历所有 3 组特征组合，GridSearchCV 训练 SVR
  6. 取前 5 名模型做集成平均
  7. 全区预测并输出 Petrel 格式文件

用法：
  python scripts/predict_sand_thickness.py \
      --seismic data/target/H6-2 \
      --wells scripts/output/well_data_preprocess_20260701_160047/well_horizon_processed.xlsx \
      --pseudo-wells scripts/output/make_pesudo_sample_20260701_160624/H6_2_optimized_pseudo_wells.csv \
      --surface H6-2
"""

import argparse
import os
import random
import sys
import time
import warnings
from datetime import datetime
from itertools import combinations
from math import comb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")

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
        "num_points": 5,
    },
    # -- 特征相关性分组
    "corr_threshold": 0.9,
    # -- SVR 组合选择
    "n_select_groups": 3,            # 每次选几组特征
    "top_models": 5,                 # 集成模型数
    "grid_search_n_jobs": 1,         # Windows 下避免 joblib 子进程启动/权限问题
    # -- SVR 参数网格
    "param_grid": [
        {"C": [0.01, 0.1, 1], "gamma": ["scale", 0.001, 0.01, 0.1],
         "epsilon": [0.1, 0.2, 0.5], "kernel": ["rbf"]},
        {"C": [0.01, 0.1, 1, 10], "epsilon": [0.1, 0.2, 0.5], "kernel": ["linear"]},
    ],
    # -- 动态数据增强
    "augmentation": {
        "target_samples_per_bin": 10,
        "noise_factor": 0.03,
        "thickness_bins": [0, 1, 13.75, 27.5, np.inf],
    },
    # -- 样本权重
    "sample_weights": {
        "real": 5.0,
        "real_augmented": 3.0,
        "pseudo_sampled": 1.5,
        "pseudo_original": 1.0,
    },
    # -- 可视化
    "class_thresholds": [1.0, 13.75],
    "vrange": (0, 20),
}
# ================================================================

from src.data_utils import (
    extract_seismic_attributes_for_wells,
    identify_attributes,
    parse_petrel_file,
    preprocess_features,
)
from src.feature_selection import group_features_by_correlation, select_features_from_groups
from src.visualization import visualize_attribute_map

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
    p = argparse.ArgumentParser(description="SVR 集成预测砂厚")
    p.add_argument("--seismic", required=True, help="地震属性 Surface 文件路径")
    p.add_argument("--wells", required=True, help="井点 xlsx 文件路径")
    p.add_argument("--pseudo-wells", required=True, help="虚拟井 csv 文件路径")
    p.add_argument("--surface", required=True, help="目标层位名称")
    p.add_argument("--corr-threshold", type=float, default=CFG["corr_threshold"], help="相关性分组阈值")
    p.add_argument("--top-models", type=int, default=CFG["top_models"], help="集成模型数量")
    return p.parse_args()


def dynamic_data_augmentation(X_real, y_real, X_pseudo, y_pseudo,
                              target_samples_per_bin=None, noise_factor=None):
    """动态数据增强：按砂厚区间均衡样本分布"""
    print("\n=== 动态数据增强策略 ===")
    thickness_bins = CFG["augmentation"]["thickness_bins"]
    bin_labels = ["0-1m", "1-13.75m", "13.75-27.5m", ">27.5m"]

    real_bin_counts, real_bin_masks = [], []
    print("真实井样本分布分析:")
    for i in range(len(thickness_bins) - 1):
        mask = (y_real >= thickness_bins[i]) & (y_real < thickness_bins[i + 1])
        real_bin_counts.append(np.sum(mask))
        real_bin_masks.append(mask)
        print(f"  {bin_labels[i]}: {real_bin_counts[-1]} 个真实样本")

    pseudo_bin_counts, pseudo_bin_masks = [], []
    print("\n虚拟井样本分布分析:")
    for i in range(len(thickness_bins) - 1):
        mask = (y_pseudo >= thickness_bins[i]) & (y_pseudo < thickness_bins[i + 1])
        pseudo_bin_counts.append(np.sum(mask))
        pseudo_bin_masks.append(mask)
        print(f"  {bin_labels[i]}: {pseudo_bin_counts[-1]} 个虚拟样本")

    X_augmented = X_real.copy()
    y_augmented = y_real.copy()
    augmentation_sources = ["real"] * len(y_real)

    all_features = np.vstack([X_real, X_pseudo])
    feature_stds = np.std(all_features, axis=0)

    print(f"\n动态增强决策（阈值: {target_samples_per_bin}）:")
    for i in range(len(thickness_bins) - 1):
        real_count = real_bin_counts[i]
        pseudo_count = pseudo_bin_counts[i]
        print(f"\n{bin_labels[i]} 区间: 真实={real_count}, 虚拟={pseudo_count}")

        if real_count >= target_samples_per_bin:
            print(f"  真实样本充足，无需增强")
            continue

        samples_needed = target_samples_per_bin - real_count
        print(f"  需要增强 {samples_needed} 个样本")

        if real_count > 0:
            X_real_bin = X_real[real_bin_masks[i]]
            y_real_bin = y_real[real_bin_masks[i]]
            for _ in range(samples_needed):
                base_idx = np.random.randint(0, len(X_real_bin))
                base_x, base_y = X_real_bin[base_idx].copy(), y_real_bin[base_idx]
                if base_y <= 1:
                    af = noise_factor * 0.5
                elif base_y <= 13.75:
                    af = noise_factor * 1.0
                else:
                    af = noise_factor * 1.5
                new_x = base_x + np.random.normal(0, feature_stds * af)
                if i == 0:
                    new_y = np.clip(base_y + np.random.uniform(-0.1, 0.1), 0, 0.99)
                elif i == 1:
                    new_y = np.clip(base_y + np.random.uniform(-0.5, 0.5), 1, 13.75)
                elif i == 2:
                    new_y = np.clip(base_y + np.random.uniform(-1.0, 1.0), 13.75, 27.5)
                else:
                    new_y = max(27.5, base_y + np.random.uniform(-2.0, 2.0))
                X_augmented = np.vstack([X_augmented, new_x.reshape(1, -1)])
                y_augmented = np.append(y_augmented, new_y)
                augmentation_sources.append("real_augmented")
        elif pseudo_count > 0:
            X_pseudo_bin = X_pseudo[pseudo_bin_masks[i]]
            y_pseudo_bin = y_pseudo[pseudo_bin_masks[i]]
            n_sample = min(samples_needed, len(X_pseudo_bin))
            indices = np.random.choice(len(X_pseudo_bin), size=n_sample,
                                       replace=samples_needed > len(X_pseudo_bin))
            for idx in indices:
                new_x = X_pseudo_bin[idx] + np.random.normal(0, feature_stds * noise_factor * 0.3)
                X_augmented = np.vstack([X_augmented, new_x.reshape(1, -1)])
                y_augmented = np.append(y_augmented, y_pseudo_bin[idx])
                augmentation_sources.append("pseudo_sampled")
        else:
            print(f"  警告: 该区间既无真实样本也无虚拟样本，跳过增强")

    print(f"\n数据增强完成: {len(y_real)} 真实 -> {len(y_augmented)} 总样本 (+{len(y_augmented) - len(y_real)})")
    for src in set(augmentation_sources):
        print(f"  {src}: {augmentation_sources.count(src)}")
    return X_augmented, y_augmented, augmentation_sources


def plot_svr_model_selection_summary(
    model_results, selected_count, save_path, display_count=10
):
    """绘制按 CV R^2 排名的 SVR 模型筛选摘要。"""
    ranked_models = model_results[: min(display_count, len(model_results))]
    if not ranked_models:
        raise ValueError("没有成功训练的 SVR 模型，无法绘制筛选摘要")

    selected_count = min(selected_count, len(ranked_models))
    scores = np.array([result["cv_r2"] for result in ranked_models])
    labels = [
        f"M{rank + 1:02d} | " + " + ".join(result["selected_features"])
        for rank, result in enumerate(ranked_models)
    ]
    colors = [
        "#4C78A8" if rank < selected_count else "#BAB0AC"
        for rank in range(len(ranked_models))
    ]

    fig, ax = plt.subplots(figsize=(16, 9))
    y_positions = np.arange(len(ranked_models))
    bars = ax.barh(y_positions, scores, color=colors, edgecolor="white", height=0.72)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)

    score_span = max(scores.max() - scores.min(), 0.1)
    label_offset = score_span * 0.02
    for bar, score in zip(bars, scores):
        if score >= 0:
            x_value, alignment = score + label_offset, "left"
        else:
            x_value, alignment = score - label_offset, "right"
        ax.text(
            x_value, bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}", va="center", ha=alignment, fontsize=10,
        )

    if selected_count:
        cutoff_score = scores[selected_count - 1]
        ax.axvline(
            cutoff_score, color="#E45756", linestyle="--", linewidth=1.8,
            label=f"Top-{selected_count} 截止 CV R^2={cutoff_score:.3f}",
        )
        if selected_count < len(ranked_models):
            ax.axhline(selected_count - 0.5, color="#E45756", linestyle=":", linewidth=1.5)

    ax.set_xlabel("交叉验证 CV R^2", fontsize=12)
    ax.set_ylabel("模型排名与特征组合", fontsize=12)
    ax.set_title(
        f"SVR 模型筛选摘要（前 {len(ranked_models)} 名）", fontsize=16
    )
    ax.grid(axis="x", alpha=0.25)
    legend_handles = [
        Patch(facecolor="#4C78A8", label=f"入选集成 Top-{selected_count}"),
        Patch(facecolor="#BAB0AC", label="未入选模型"),
    ]
    if selected_count:
        legend_handles.append(
            Line2D(
                [0], [0], color="#E45756", linestyle="--",
                label=f"第 {selected_count} 名分数截止线",
            )
        )
    ax.legend(handles=legend_handles, loc="lower right", fontsize=10)
    plt.subplots_adjust(left=0.42, right=0.96, top=0.92, bottom=0.10)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    SURFACE_NAME = args.surface
    output_dir, figures_dir, data_tmp_dir = setup_output_dir("predict_sand_thickness")

    print(f"输出目录: {output_dir}")
    print(f"图件目录: {figures_dir}")

    # ================================================================
    # 步骤 1: 加载地震数据 & 属性预处理
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 1: 加载地震数据 & 属性预处理")
    print("=" * 60)

    data_seismic_attr = parse_petrel_file(args.seismic)
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
    # 步骤 2: 加载真实井 & 虚拟井
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 2: 加载真实井 & 虚拟井")
    print("=" * 60)

    data_well_position = pd.read_excel(args.wells)
    data_well_purpose_surface_position = (
        data_well_position[data_well_position["Surface"] == SURFACE_NAME]
        .replace(-999, np.nan)
        .dropna(subset=["Sand Thickness"])
        .reset_index(drop=True)
    )
    print(f"层位 {SURFACE_NAME} 井点数量: {len(data_well_purpose_surface_position)}")

    w = CFG["well_attr_extract"]
    well_attr = extract_seismic_attributes_for_wells(
        well_data=data_well_purpose_surface_position,
        seismic_data=data_seismic_attr_processed,
        max_distance=w["max_distance"], num_points=w["num_points"],
    )
    well_attr.to_excel(os.path.join(data_tmp_dir, "wells_attr.xlsx"), index=False)
    print(f"井点地震属性提取完成，共 {len(well_attr)} 口井")

    pseudo_wells = pd.read_csv(args.pseudo_wells)
    print(f"虚拟井数据导入完成，共 {len(pseudo_wells)} 个虚拟井点")

    if "Predicted_Sand_Thickness" in pseudo_wells.columns:
        pseudo_thickness_col = "Predicted_Sand_Thickness"
    elif "Mean_Pred" in pseudo_wells.columns:
        pseudo_thickness_col = "Mean_Pred"
    else:
        possible = [c for c in pseudo_wells.columns if "thick" in c.lower() or "pred" in c.lower()]
        if possible:
            pseudo_thickness_col = possible[0]
        else:
            raise ValueError("无法找到虚拟井砂厚列")
    print(f"虚拟井砂厚列: {pseudo_thickness_col}")
    print(f"  范围: {pseudo_wells[pseudo_thickness_col].min():.2f} ~ {pseudo_wells[pseudo_thickness_col].max():.2f} m")

    # ================================================================
    # 步骤 3: 特征自相关性分组
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 3: 特征自相关性分组")
    print("=" * 60)

    feature_groups, correlation_matrix = group_features_by_correlation(
        data=data_seismic_attr_processed,
        feature_columns=attribute_names_processed,
        correlation_threshold=args.corr_threshold,
        verbose=True,
    )
    print(f"特征分组完成，共 {len(feature_groups)} 组")

    # ================================================================
    # 步骤 4: 准备训练数据 & 数据增强
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 4: 准备训练数据 & 数据增强")
    print("=" * 60)

    common_features = [c for c in attribute_names_processed if c in pseudo_wells.columns]
    missing = [c for c in attribute_names_processed if c not in pseudo_wells.columns]
    if missing:
        print(f"警告: 虚拟井缺少特征: {missing}")
        feature_groups = [[f for f in g if f in common_features] for g in feature_groups]
        feature_groups = [g for g in feature_groups if g]
        print(f"更新后特征组数: {len(feature_groups)}")

    X_real = well_attr[common_features].values
    y_real = well_attr["Sand Thickness"].values
    X_pseudo = pseudo_wells[common_features].values
    y_pseudo = pseudo_wells[pseudo_thickness_col].values

    a = CFG["augmentation"]
    X_real_aug, y_real_aug, aug_sources = dynamic_data_augmentation(
        X_real, y_real, X_pseudo, y_pseudo,
        target_samples_per_bin=a["target_samples_per_bin"],
        noise_factor=a["noise_factor"],
    )

    X_combined = np.vstack([X_real_aug, X_pseudo])
    y_combined = np.concatenate([y_real_aug, y_pseudo])

    sw = CFG["sample_weights"]
    sample_weights = np.concatenate([
        np.ones(len(X_real)) * sw["real"],
        np.ones(aug_sources.count("real_augmented")) * sw["real_augmented"],
        np.ones(aug_sources.count("pseudo_sampled")) * sw["pseudo_sampled"],
        np.ones(len(X_pseudo)) * sw["pseudo_original"],
    ])
    print(f"总训练样本: {len(X_combined)}, 目标范围: {y_combined.min():.2f} ~ {y_combined.max():.2f}")

    # ================================================================
    # 步骤 5: 遍历特征组合训练 SVR
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 5: 遍历特征组合训练 SVR")
    print("=" * 60)

    n_select = min(CFG["n_select_groups"], len(feature_groups))
    all_combinations = list(combinations(range(len(feature_groups)), n_select))
    total_comb = len(all_combinations)

    param_grid = CFG["param_grid"]
    print(f"总组合数: {total_comb}, 参数网格: RBF(36) + Linear(12) = 48 个模型/组合")
    print(f"预计训练: {total_comb} × 48 = {total_comb * 48} 个 SVR 模型")

    np.random.seed(42)
    random.seed(42)
    model_results = []
    start_time = time.time()

    for i, combination in enumerate(all_combinations):
        print(f"\n模型 {i + 1}/{total_comb}: 组合 {combination}")
        selected_features = select_features_from_groups(
            feature_groups, combination, random_seed=42 + i
        )
        feature_indices = [common_features.index(f) for f in selected_features if f in common_features]
        if len(feature_indices) != len(selected_features):
            print(f"  跳过: 部分特征不在数据中")
            continue

        X_train = X_combined[:, feature_indices]
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        try:
            cv_folds = max(2, min(3, len(X_train) // 3))
            grid_search = GridSearchCV(
                SVR(), param_grid, cv=cv_folds, scoring="r2",
                n_jobs=CFG["grid_search_n_jobs"], return_train_score=True,
            )
            grid_search.fit(X_train_scaled, y_combined, sample_weight=sample_weights)

            best_svr = grid_search.best_estimator_
            best_cv = grid_search.best_score_
            train_pred = best_svr.predict(X_train_scaled)
            train_r2 = r2_score(y_combined, train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_combined, train_pred))
            overfit = train_r2 - best_cv

            model_results.append({
                "combination": combination,
                "selected_features": selected_features,
                "best_params": grid_search.best_params_,
                "cv_r2": best_cv, "train_r2": train_r2,
                "train_rmse": train_rmse, "overfitting_score": overfit,
                "model": best_svr, "scaler": scaler,
                "feature_indices": feature_indices,
            })
            print(f"  CV R^2={best_cv:.4f}, Train R^2={train_r2:.4f}, RMSE={train_rmse:.4f}")
            if overfit > 0.5:
                print(f"  警告: 过拟合差异 {overfit:.3f}")

        except Exception as e:
            print(f"  训练失败: {e}")

        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_t = elapsed / (i + 1)
            remaining = avg_t * (total_comb - i - 1)
            print(f"\n进度: {i + 1}/{total_comb}, 已用时: {elapsed:.0f}s, 预计剩余: {remaining:.0f}s")

    total_time = time.time() - start_time
    print(f"\n训练完成: {len(model_results)}/{total_comb} 成功, 总用时: {total_time:.0f}s")

    # ================================================================
    # 步骤 6: 选择最佳模型 & 全区预测
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 6: Top-{} 模型集成预测".format(args.top_models))
    print("=" * 60)

    model_results.sort(key=lambda x: x["cv_r2"], reverse=True)
    top_models = model_results[:args.top_models]

    for i, r in enumerate(top_models):
        print(f"\n模型 {i + 1}:")
        print(f"  特征: {r['selected_features']}")
        print(f"  参数: {r['best_params']}")
        print(f"  CV R^2={r['cv_r2']:.4f}, Train R^2={r['train_r2']:.4f}")

    model_selection_figure_path = os.path.join(
        figures_dir, "svr_model_selection_summary.png"
    )
    plot_svr_model_selection_summary(
        model_results,
        selected_count=len(top_models),
        save_path=model_selection_figure_path,
        display_count=10,
    )
    print(f"SVR 模型筛选摘要图已保存: {model_selection_figure_path}")

    seismic_data = data_seismic_attr_processed.copy()
    seismic_features = seismic_data[common_features].fillna(seismic_data[common_features].mean())

    for i, r in enumerate(top_models):
        X_s = seismic_features.iloc[:, r["feature_indices"]].values
        X_s_scaled = r["scaler"].transform(X_s)
        pred = np.maximum(r["model"].predict(X_s_scaled), 0)
        seismic_data[f"SVR_Model_{i + 1}_Prediction"] = pred

    ensemble_pred = np.mean(
        [seismic_data[f"SVR_Model_{j + 1}_Prediction"].values for j in range(len(top_models))], axis=0
    )
    seismic_data["SVR_Ensemble_Prediction"] = ensemble_pred
    print(f"\n集成预测: {ensemble_pred.min():.2f} ~ {ensemble_pred.max():.2f} m, 均值: {ensemble_pred.mean():.2f}")

    # 保存 CSV
    csv_path = os.path.join(output_dir, "svr_predictions_all_models.csv")
    seismic_data.to_csv(csv_path, index=False)
    print(f"预测结果已保存: {csv_path}")

    # Petrel txt
    base_coords = seismic_data[["X", "Y"]].copy()
    for i in range(len(top_models)):
        col = f"SVR_Model_{i + 1}_Prediction"
        petrel_df = base_coords.copy()
        petrel_df["Sand Thickness"] = seismic_data[col]
        petrel_df.to_csv(os.path.join(output_dir, f"SVR_Model_{i + 1}_Prediction.txt"),
                         sep=" ", index=False, header=True, float_format="%.6f")
    ensemble_df = base_coords.copy()
    ensemble_df["Sand Thickness"] = ensemble_pred
    ensemble_df.to_csv(os.path.join(output_dir, "SVR_Ensemble_Prediction.txt"),
                       sep=" ", index=False, header=True, float_format="%.6f")
    print("Petrel 格式文件已生成")

    # 模型结果摘要
    summary_rows = []
    for i, r in enumerate(model_results):
        summary_rows.append({
            "model_rank": i + 1,
            "group_combination": str(r["combination"]),
            "selected_features": str(r["selected_features"]),
            "cv_r2": r["cv_r2"], "train_r2": r["train_r2"],
            "train_rmse": r["train_rmse"],
            "best_params": str(r["best_params"]),
        })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(output_dir, "svr_model_results_summary.csv"), index=False
    )

    # 预测统计
    avg_cv = np.mean([r["cv_r2"] for r in top_models])
    stats_rows = []
    for i, r in enumerate(top_models):
        col = f"SVR_Model_{i + 1}_Prediction"
        p = seismic_data[col]
        stats_rows.append({
            "Model": f"Model_{i + 1}", "Features": str(r["selected_features"]),
            "CV_R2": r["cv_r2"], "Min": p.min(), "Max": p.max(),
            "Mean": p.mean(), "Std": p.std(), "Median": p.median(),
            ">1m": (p > 1).sum(), ">13.75m": (p > 13.75).sum(), ">27.5m": (p > 27.5).sum(),
        })
    stats_rows.append({
        "Model": "Ensemble", "Features": "Top-{} avg".format(args.top_models),
        "CV_R2": avg_cv, "Min": ensemble_pred.min(), "Max": ensemble_pred.max(),
        "Mean": ensemble_pred.mean(), "Std": ensemble_pred.std(), "Median": np.median(ensemble_pred),
        ">1m": (ensemble_pred > 1).sum(), ">13.75m": (ensemble_pred > 13.75).sum(),
        ">27.5m": (ensemble_pred > 27.5).sum(),
    })
    pd.DataFrame(stats_rows).to_csv(
        os.path.join(output_dir, "prediction_statistics.csv"), index=False, encoding="utf-8-sig"
    )

    # ================================================================
    # 步骤 7: 可视化
    # ================================================================
    print("\n" + "=" * 60)
    print("步骤 7: 可视化预测结果")
    print("=" * 60)

    viz_args = dict(
        real_wells=well_attr, target_column="Sand Thickness",
        output_dir=figures_dir, class_thresholds=CFG["class_thresholds"],
        figsize=(14, 14), point_size=10, well_size=60,
        vrange=CFG["vrange"], cmap="viridis",
    )
    for i, r in enumerate(top_models):
        col = f"SVR_Model_{i + 1}_Prediction"
        visualize_attribute_map(
            data_points=seismic_data, attribute_name=col,
            attribute_label=f"Model {i + 1} CV R^2={r['cv_r2']:.3f}",
            filename_prefix=f"svr_model_{i + 1}_prediction", **viz_args,
        )

    visualize_attribute_map(
        data_points=seismic_data, attribute_name="SVR_Ensemble_Prediction",
        attribute_label=f"SVR Ensemble avg CV R^2={avg_cv:.3f}",
        filename_prefix="svr_ensemble_prediction", **viz_args,
    )

    print("\n" + "=" * 60)
    print(f"全部完成！输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
