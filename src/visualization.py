import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize_model_results(
    y_test,
    y_pred,
    y_test_zero,
    y_pred_is_zero,
    threshold_zero=0.1,
    output_dir="output",
    filename_prefix="svr_model",
):
    """
    可视化模型性能并保存结果

    参数:
        y_test: 测试集真实值
        y_pred: 测试集预测值
        y_test_zero: 测试集中真实值是否为零的布尔数组
        y_pred_is_zero: 测试集中预测值是否为零的布尔数组
        threshold_zero: 零值阈值
        output_dir: 输出目录
        filename_prefix: 文件名前缀
    """
    # 创建真值vs预测值散点图
    plt.figure(figsize=(10, 8))

    # 绘制散点图，区分"砂厚=0"和"砂厚>0"的样本
    plt.scatter(
        y_test[~y_test_zero],
        y_pred[~y_test_zero],
        c="blue",
        alpha=0.6,
        label="砂厚 > 0",
    )
    plt.scatter(
        y_test[y_test_zero], y_pred[y_test_zero], c="red", alpha=0.6, label="砂厚 ≈ 0"
    )

    # 绘制对角线 (y=x)
    max_val = max(np.max(y_test), np.max(y_pred))
    plt.plot([0, max_val], [0, max_val], "k--", label="理想预测")

    # 绘制阈值线
    plt.axhline(
        y=threshold_zero, color="green", linestyle=":", label=f"阈值 = {threshold_zero}"
    )
    plt.axvline(x=threshold_zero, color="green", linestyle=":")

    plt.xlabel("真实砂厚")
    plt.ylabel("预测砂厚")
    plt.title("SVR回归：真实值 vs 预测值")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 保存图像
    plt.savefig(
        os.path.join(output_dir, f"{filename_prefix}_true_vs_pred.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()

    # 创建误差直方图
    errors = y_pred - y_test
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=30, alpha=0.7, color="blue")
    plt.axvline(x=0, color="red", linestyle="--")
    plt.xlabel("预测误差 (预测值 - 真实值)")
    plt.ylabel("频数")
    plt.title("SVR回归预测误差分布")
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(output_dir, f"{filename_prefix}_error_distribution.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()

    # 保存预测结果
    results_df = pd.DataFrame(
        {
            "True_Value": y_test,
            "Predicted_Value": y_pred,
            "Is_Zero_True": y_test_zero,
            "Is_Zero_Pred": y_pred_is_zero,
            "Error": errors,
        }
    )
    results_df.to_csv(
        os.path.join(output_dir, f"{filename_prefix}_prediction_results.csv"),
        index=False,
    )


def visualize_predictions(
    prediction_results,
    output_dir="output",
    filename_prefix="predicted",
    threshold_zero=0.1,
):
    """
    可视化预测结果

    参数:
        prediction_results (DataFrame): 包含预测结果的数据框
        output_dir (str): 输出目录
        filename_prefix (str): 输出文件名前缀
        threshold_zero (float): 小于此值视为0
    """
    # 获取统计数据
    pred_mean = prediction_results["Predicted_Sand_Thickness"].mean()
    pred_max = prediction_results["Predicted_Sand_Thickness"].max()

    # 可视化预测结果 - 散点图
    plt.figure(figsize=(14, 12))
    scatter = plt.scatter(
        prediction_results["X"],
        prediction_results["Y"],
        c=prediction_results["Predicted_Sand_Thickness"],
        cmap="viridis",
        s=2,  # 点大小较小，因为点数可能很多
        alpha=0.7,
        vmin=0,
        vmax=min(pred_max, pred_mean * 3),  # 限制颜色范围，便于观察
    )
    plt.colorbar(scatter, label="预测砂厚")
    plt.title("砂厚预测结果空间分布")
    plt.xlabel("X坐标")
    plt.ylabel("Y坐标")
    plt.savefig(
        os.path.join(output_dir, f"{filename_prefix}_sand_thickness_map.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()

    # 可视化预测结果 - 直方图
    plt.figure(figsize=(12, 6))
    plt.hist(
        prediction_results["Predicted_Sand_Thickness"], bins=50, alpha=0.7, color="blue"
    )
    plt.axvline(
        x=threshold_zero, color="red", linestyle="--", label=f"阈值 = {threshold_zero}"
    )
    plt.xlabel("预测砂厚")
    plt.ylabel("频数")
    plt.title("砂厚预测结果分布")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(
        os.path.join(output_dir, f"{filename_prefix}_sand_thickness_histogram.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close()


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
    petrel_output_file = os.path.join(
        output_dir, f"{filename_prefix}_sand_thickness_petrel.txt"
    )
    with open(petrel_output_file, "w") as f:
        # 写入标题
        f.write("# 砂厚预测结果\n")
        f.write(f"# {' '.join(coords_columns)} Predicted_Sand_Thickness\n")

        # 写入数据
        for _, row in prediction_results.iterrows():
            coords = " ".join([f"{row[col]:.6f}" for col in coords_columns])
            f.write(f"{coords} {row['Predicted_Sand_Thickness']:.6f}\n")

    print(f"预测结果已保存为Petrel可导入的XYZ格式: {petrel_output_file}")
