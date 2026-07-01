"""
井点数据预处理脚本

功能：
  1. 读取原始井点数据 (well_horizon.xlsx)
  2. 删除指定层位和井点
  3. 删除砂厚为 -999 的无效数据
  4. 处理重复井点（保留砂厚最大值）
  5. IQR 离群值检测与剔除（3.0×IQR）
  6. 输出处理后的数据

用法：
  python scripts/well_data_preprocess.py
  python scripts/well_data_preprocess.py --input data/target/well_horizon.xlsx
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# 将项目根目录加入路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def setup_output_dir(script_name):
    """创建带时间戳的输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(PROJECT_ROOT, "scripts", "output")
    output_dir = os.path.join(output_root, f"{script_name}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="井点数据预处理")
    parser.add_argument(
        "--input",
        default=os.path.join(PROJECT_ROOT, "data", "target", "well_horizon.xlsx"),
        help="输入 xlsx 文件路径",
    )
    parser.add_argument("--sheet", default="Sand Thickness", help="Sheet 名称")
    parser.add_argument(
        "--horizons-to-delete", nargs="*", default=["P0(H83D)"], help="要删除的层位列表"
    )
    parser.add_argument(
        "--wells-to-delete", nargs="*", default=[], help="要删除的井点列表"
    )
    parser.add_argument(
        "--iqr-factor", type=float, default=3.0, help="IQR 离群值倍数（默认 3.0）"
    )
    parser.add_argument(
        "--output-name",
        default="well_horizon_processed.xlsx",
        help="输出文件名",
    )
    args = parser.parse_args()

    # 创建输出目录
    output_dir = setup_output_dir("well_data_preprocess")
    print(f"输出目录: {output_dir}")

    # ==================== 1. 读取数据 ====================
    print("\n" + "=" * 60)
    print("步骤 1: 读取数据")
    print("=" * 60)
    data_well = pd.read_excel(args.input, sheet_name=args.sheet)
    print(f"原始数据形状: {data_well.shape}")
    print(f"原始数据前 5 行:\n{data_well.head()}")

    # ==================== 2. 识别关键列 ====================
    print("\n" + "=" * 60)
    print("步骤 2: 识别关键列")
    print("=" * 60)
    xyz_columns = data_well.columns[0:3].tolist()
    surface_column = data_well.columns[9]
    well_column = data_well.columns[10]
    sand_thickness_column = data_well.columns[27]
    count_column = data_well.columns[28]

    print(f"XYZ 坐标列: {xyz_columns}")
    print(f"层位名列:   {surface_column}")
    print(f"井名列:     {well_column}")
    print(f"砂厚列:     {sand_thickness_column}")
    print(f"重复计数列: {count_column}")

    # ==================== 3. 删除指定层位和井点 ====================
    print("\n" + "=" * 60)
    print("步骤 3: 删除指定层位和井点")
    print("=" * 60)
    data_filtered = data_well.copy()
    total_removed_this_step = 0

    if args.horizons_to_delete:
        before = len(data_filtered)
        data_filtered = data_filtered[~data_filtered[surface_column].isin(args.horizons_to_delete)]
        removed = before - len(data_filtered)
        total_removed_this_step += removed
        print(f"删除层位 {args.horizons_to_delete}，移除了 {removed} 行")

    if args.wells_to_delete:
        before = len(data_filtered)
        data_filtered = data_filtered[~data_filtered[well_column].isin(args.wells_to_delete)]
        removed = before - len(data_filtered)
        total_removed_this_step += removed
        print(f"删除井点 {args.wells_to_delete}，移除了 {removed} 行")

    if total_removed_this_step == 0:
        print("无需删除的层位或井点")
    print(f"当前数据: {len(data_filtered)} 行")

    # ==================== 4. 删除砂厚为 -999 的行 ====================
    print("\n" + "=" * 60)
    print("步骤 4: 删除砂厚为 -999 的无效数据")
    print("=" * 60)
    missing_mask = data_filtered[sand_thickness_column] == -999
    missing_count = missing_mask.sum()

    if missing_count > 0:
        print(f"发现 {missing_count} 个砂厚值为 -999 的数据")
        missing_data = data_filtered[missing_mask]
        print("示例（前 5 行）：")
        for _, row in missing_data.head().iterrows():
            print(f"  井: {row[well_column]}, 层位: {row[surface_column]}, 砂厚: {row[sand_thickness_column]}")
        data_filtered = data_filtered[~missing_mask].reset_index(drop=True)
        print(f"已删除 {missing_count} 行，剩余 {len(data_filtered)} 行")
    else:
        print("未发现砂厚值为 -999 的数据")

    # ==================== 5. 处理重复数据 ====================
    print("\n" + "=" * 60)
    print("步骤 5: 处理重复井点数据")
    print("=" * 60)
    duplicate_mask = data_filtered[count_column] > 1
    duplicate_count = duplicate_mask.sum()

    if duplicate_count > 0:
        print(f"发现 {duplicate_count} 行重复计数 > 1 的数据")
        duplicates = data_filtered[duplicate_mask]
        non_duplicates = data_filtered[~duplicate_mask]

        shown = 0
        for (well, surface), group in duplicates.groupby([well_column, surface_column]):
            if shown < 3:
                print(f"  井 {well}, 层位 {surface}: {len(group)} 行, "
                      f"砂厚范围 {group[sand_thickness_column].min():.2f} ~ {group[sand_thickness_column].max():.2f}")
                shown += 1

        max_thickness_duplicates = duplicates.loc[
            duplicates.groupby([well_column, surface_column])[sand_thickness_column].idxmax()
        ]
        data_processed = pd.concat([non_duplicates, max_thickness_duplicates], ignore_index=True)
        removed = len(data_filtered) - len(data_processed)
        print(f"移除了 {removed} 行重复数据，保留砂厚最大的行")
    else:
        data_processed = data_filtered
        print("没有发现重复计数 > 1 的数据")

    # ==================== 6. 离群值处理 ====================
    print("\n" + "=" * 60)
    print("步骤 6: IQR 离群值检测与剔除")
    print("=" * 60)

    Q1 = data_processed[sand_thickness_column].quantile(0.25)
    Q3 = data_processed[sand_thickness_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - args.iqr_factor * IQR
    upper_bound = Q3 + args.iqr_factor * IQR

    print(f"Q1 (25%): {Q1:.2f} m")
    print(f"Q3 (75%): {Q3:.2f} m")
    print(f"IQR:      {IQR:.2f} m")
    print(f"离群值阈值: [{lower_bound:.2f}, {upper_bound:.2f}] m ({args.iqr_factor}×IQR)")

    outliers_mask = (data_processed[sand_thickness_column] < lower_bound) | (
        data_processed[sand_thickness_column] > upper_bound
    )
    outliers_count = outliers_mask.sum()

    if outliers_count > 0:
        print(f"\n发现 {outliers_count} 个离群值:")
        outliers = data_processed[outliers_mask]
        for _, row in outliers.iterrows():
            print(f"  井: {row[well_column]}, 层位: {row[surface_column]}, 砂厚: {row[sand_thickness_column]:.2f} m")
        data_processed = data_processed[~outliers_mask].reset_index(drop=True)
        print(f"\n已删除 {outliers_count} 个离群值，剩余 {len(data_processed)} 行")
    else:
        print("\n未发现离群值")

    # ==================== 7. 最终统计 ====================
    print("\n" + "=" * 60)
    print("步骤 7: 最终处理结果统计")
    print("=" * 60)
    print(f"原始数据:     {len(data_well)} 行")
    print(f"处理后数据:   {len(data_processed)} 行")
    print(f"数据保留率:   {len(data_processed) / len(data_well) * 100:.1f}%")
    print(f"不同井点数:   {len(data_processed[well_column].unique())}")
    print(f"不同层位数:   {len(data_processed[surface_column].unique())}")

    well_counts = data_processed[well_column].value_counts()
    print(f"\n每井数据量统计:")
    print(f"  最多: {well_counts.max()} 行 (井: {well_counts.idxmax()})")
    print(f"  最少: {well_counts.min()} 行 (井: {well_counts.idxmin()})")
    print(f"  平均: {well_counts.mean():.1f} 行")

    print(f"\n砂厚分布统计:")
    print(data_processed[sand_thickness_column].describe())

    zero_count = (data_processed[sand_thickness_column] == 0).sum()
    pos_count = (data_processed[sand_thickness_column] > 0).sum()
    print(f"砂厚 = 0:  {zero_count} 个 ({zero_count / len(data_processed) * 100:.1f}%)")
    print(f"砂厚 > 0:  {pos_count} 个 ({pos_count / len(data_processed) * 100:.1f}%)")

    # ==================== 8. 保存结果 ====================
    print("\n" + "=" * 60)
    print("步骤 8: 保存处理结果")
    print("=" * 60)

    output_path = os.path.join(output_dir, args.output_name)
    data_processed.to_excel(output_path, index=False)
    print(f"处理后数据已保存: {output_path}")

    log_path = os.path.join(output_dir, "preprocess_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"井点数据预处理日志\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入文件: {args.input}\n")
        f.write(f"原始行数: {len(data_well)}\n")
        f.write(f"最终行数: {len(data_processed)}\n")
        f.write(f"保留率:   {len(data_processed) / len(data_well) * 100:.1f}%\n")
        f.write(f"删除层位: {args.horizons_to_delete}\n")
        f.write(f"删除井点: {args.wells_to_delete}\n")
        f.write(f"IQR 倍数: {args.iqr_factor}\n")
        f.write(f"离群值阈值: [{lower_bound:.2f}, {upper_bound:.2f}]\n")
        f.write(f"删除离群值: {outliers_count} 行\n")
        f.write(f"输出文件: {output_path}\n")
    print(f"处理日志已保存: {log_path}")

    print("\n" + "=" * 60)
    print("预处理完成！")
    print(f"输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
