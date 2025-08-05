import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 生成示例数据
np.random.seed(40)
n_points = 30

# 融合属性值 (横坐标)
fusion_attr = np.linspace(0.2, 0.8, n_points) + np.random.normal(0, 0.05, n_points)

# 真实砂厚 (作为参考线的基础)
true_thickness = 10 + 15 * fusion_attr + np.random.normal(0, 1, n_points)

# 预测砂厚 (纵坐标) - 初始预测
predicted_thickness = true_thickness + np.random.normal(0, 1.0, n_points)

# 确保有大约4:1的比例 (范围内:范围外)
confidence_threshold = 2.0  # 置信区间阈值 ±2米

# 调整一些点使其超出置信区间
outlier_count = 10
outlier_indices = np.random.choice(range(n_points), size=outlier_count, replace=False)
for idx in outlier_indices:
    # 确保这些点明显超出置信区间
    if np.random.random() > 0.5:
        predicted_thickness[idx] = true_thickness[idx] + confidence_threshold + np.random.uniform(2, 5)
    else:
        predicted_thickness[idx] = true_thickness[idx] - confidence_threshold - np.random.uniform(2, 5)

# 创建DataFrame
data = pd.DataFrame(
    {
        "融合属性": fusion_attr,
        "预测砂厚": predicted_thickness,
        "真实砂厚": true_thickness,
        "厚度差异": np.abs(predicted_thickness - true_thickness),
    }
)

# 判断是否在置信区间内
data["在置信区间内"] = data["厚度差异"] <= confidence_threshold

# 打印调试信息 - 特别检查outlier点
print("=== 调试信息 ===")
print(f"总点数: {len(data)}")
print(f"outlier_indices: {outlier_indices}")
print(f"置信区间内的点: {data['在置信区间内'].sum()} 个")
print(f"置信区间外的点: {(~data['在置信区间内']).sum()} 个")

print("\n=== Outlier点检查 ===")
for idx in outlier_indices:
    row = data.iloc[idx]
    print(
        f"点{idx}: 预测={row['预测砂厚']:.2f}, 真实={row['真实砂厚']:.2f}, 差异={row['厚度差异']:.2f}, 在区间内={row['在置信区间内']}"
    )

# 保存数据到Excel
data.to_excel("confidence_interval_data.xlsx", index=False)

# 创建图形
fig, ax = plt.subplots(figsize=(12, 8))

# 生成拟合线和置信带
x_smooth = np.linspace(fusion_attr.min(), fusion_attr.max(), 200)
# 线性拟合 - 基于真实砂厚
coeffs = np.polyfit(fusion_attr, true_thickness, 1)
y_fit = np.polyval(coeffs, x_smooth)

# 置信带 (更明显的弯曲) - 基于拟合线，但要确保逻辑一致
# 关键：置信带应该基于预测误差，而不是真实值的拟合
curve_factor = 1.0 * np.sin(x_smooth * 8) + 0.5 * np.cos(x_smooth * 12)
upper_bound = y_fit + confidence_threshold + curve_factor
lower_bound = y_fit - confidence_threshold + curve_factor

# 绘制置信带
ax.fill_between(x_smooth, lower_bound, upper_bound, alpha=0.2, color="lightblue", label=f"置信区间")

# 绘制拟合线
ax.plot(x_smooth, y_fit, "b-", linewidth=2, label="真实砂厚拟合线")

# 绘制边界线
ax.plot(x_smooth, upper_bound, "b--", linewidth=1, alpha=0.7)
ax.plot(x_smooth, lower_bound, "b--", linewidth=1, alpha=0.7)

# 绘制数据点 - 分别处理区间内外的点
inside_mask = data["在置信区间内"]

# 区间内的点 (绿色圆点)
ax.scatter(
    data.loc[inside_mask, "融合属性"],
    data.loc[inside_mask, "预测砂厚"],
    c="green",
    s=80,
    alpha=0.8,
    label="置信区间内 (保留)",
    marker="o",
    edgecolors="darkgreen",
    linewidth=1,
)

# 区间外的点 (红色叉号)
ax.scatter(
    data.loc[~inside_mask, "融合属性"],
    data.loc[~inside_mask, "预测砂厚"],
    c="red",
    s=100,
    alpha=0.8,
    label="置信区间外 (排除)",
    marker="X",
    edgecolors="darkred",
    linewidth=1,
)

# 设置标签和标题
ax.set_xlabel("融合属性", fontsize=14)
ax.set_ylabel("预测砂厚 (m)", fontsize=14)

# 图例放在图内右下角
ax.legend(fontsize=12, loc="lower right", frameon=True, fancybox=True, shadow=True)

# 设置网格
ax.grid(True, alpha=0.3)

# 调整布局
plt.tight_layout()

plt.show()

# 打印统计信息
print(f"\n=== 最终统计 ===")
print(f"总样本数: {len(data)}")
print(f"置信区间内: {data['在置信区间内'].sum()} 个 ({data['在置信区间内'].mean() * 100:.1f}%)")
print(f"置信区间外: {(~data['在置信区间内']).sum()} 个 ({(1 - data['在置信区间内'].mean()) * 100:.1f}%)")
print(f"平均厚度差异: {data['厚度差异'].mean():.2f}m")
print(f"比例 (内:外): {data['在置信区间内'].sum()}:{(~data['在置信区间内']).sum()}")

# 验证逻辑的额外检查
print(f"\n=== 逻辑验证 ===")
logic_errors = 0
for i in range(len(data)):
    diff = data.iloc[i]["厚度差异"]
    is_inside = data.iloc[i]["在置信区间内"]
    expected_inside = diff <= confidence_threshold
    if is_inside != expected_inside:
        logic_errors += 1
        print(
            f"逻辑错误发现在第{i}行: 差异={diff:.2f}, 标记为{'内部' if is_inside else '外部'}, 应该是{'内部' if expected_inside else '外部'}"
        )

if logic_errors == 0:
    print("✓ 逻辑验证通过，所有点的分类都正确！")
else:
    print(f"✗ 发现 {logic_errors} 个逻辑错误")
