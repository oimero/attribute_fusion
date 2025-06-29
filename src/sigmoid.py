import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


class SigmoidModel:
    """
    智能Sigmoid拟合模型

    支持自动检测PC值与地质类型关系，智能添加虚拟点稳定拟合过程。

    Attributes:
    -----------
    data : pd.DataFrame
        原始输入数据
    feature_columns : list
        特征列名列表
    target_column : str
        目标变量列名
    fit_params : np.array or None
        拟合参数 [L, k, x0]
    r2_score : float or None
        模型R²评分
    current_data : pd.DataFrame or None
        包含虚拟点的当前工作数据
    """

    def __init__(self, data, feature_columns, target_column):
        """
        初始化Sigmoid模型

        Parameters:
        -----------
        data : pd.DataFrame
            输入数据，必须包含特征列和目标列
        feature_columns : list
            特征列名列表，通常为PCA组件['PC1', 'PC2', ...]
        target_column : str
            目标变量列名，如'Sand Thickness'

        Raises:
        -------
        ValueError
            当数据中缺少必要列时抛出异常
        """
        self.data = data.copy()
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.fit_params = None
        self.r2_score = None
        self.current_data = None  # 添加虚拟点后的数据

        # 检查必要的列是否存在
        missing_cols = [col for col in feature_columns + [target_column] if col not in data.columns]
        if missing_cols:
            raise ValueError(f"数据中缺少以下列: {missing_cols}")

    @staticmethod
    def sigmoid(x, L, k, x0):
        """
        标准三参数Sigmoid函数

        Parameters:
        -----------
        x : array-like
            输入变量
        L : float
            最大渐近值，表示砂厚的理论上限
        k : float
            增长率，正值表示正向增长，负值表示负向增长
        x0 : float
            中点位置，Sigmoid函数的拐点

        Returns:
        --------
        array-like
            Sigmoid函数值，范围在[0, L]之间

        Notes:
        ------
        函数形式: f(x) = L / (1 + exp(-k * (x - x0)))
        """
        return L / (1 + np.exp(-k * (x - x0)))

    def auto_detect_pc_geology_relationship(self, primary_feature="PC1", threshold_percentile=25):
        """
        自动检测PC值与地质类型的关系

        通过分析PC值的分布与砂厚的关系，自动判断低PC值和高PC值分别对应
        泥岩还是砂岩，避免虚拟点添加错误。

        Parameters:
        -----------
        primary_feature : str, default="PC1"
            用于分析的主要特征名称
        threshold_percentile : float, default=25
            用于划分低值和高值区间的百分位数阈值

        Returns:
        --------
        dict
            包含关系映射的字典
            - 'low_pc_type': str, 低PC值对应的地质类型 ('mud' 或 'sand')
            - 'high_pc_type': str, 高PC值对应的地质类型 ('mud' 或 'sand')
            - 'low_threshold': float, 低值区间阈值
            - 'high_threshold': float, 高值区间阈值
            - 'low_avg_thickness': float, 低PC值区间平均砂厚
            - 'high_avg_thickness': float, 高PC值区间平均砂厚

        Notes:
        ------
        分析逻辑：
        1. 计算指定百分位数的PC值阈值
        2. 比较低PC值区间和高PC值区间的平均砂厚
        3. 砂厚较小的区间判定为泥岩，砂厚较大的区间判定为砂岩
        """
        # 计算低PC1值和高PC1值区间的平均砂厚
        pc_values = self.data[primary_feature]
        sand_thickness = self.data[self.target_column]

        low_threshold = np.percentile(pc_values, threshold_percentile)
        high_threshold = np.percentile(pc_values, 100 - threshold_percentile)

        # 低PC1区间的平均砂厚
        low_pc_mask = pc_values <= low_threshold
        low_pc_avg_thickness = sand_thickness[low_pc_mask].mean()

        # 高PC1区间的平均砂厚
        high_pc_mask = pc_values >= high_threshold
        high_pc_avg_thickness = sand_thickness[high_pc_mask].mean()

        print(f"PC值与地质类型关系分析({primary_feature}):")
        print(f"  低PC值区间({primary_feature} ≤ {low_threshold:.2f}): 平均砂厚 {low_pc_avg_thickness:.2f}m")
        print(f"  高PC值区间({primary_feature} ≥ {high_threshold:.2f}): 平均砂厚 {high_pc_avg_thickness:.2f}m")

        # 判断关系
        if low_pc_avg_thickness < high_pc_avg_thickness:
            # 标准关系：低PC1=泥岩，高PC1=砂岩
            relationship = {"low_pc_type": "mud", "high_pc_type": "sand", "relationship_type": "standard"}
            print("  → 检测到标准关系：低PC值=泥岩，高PC值=砂岩")
        else:
            # 反向关系：低PC1=砂岩，高PC1=泥岩
            relationship = {"low_pc_type": "sand", "high_pc_type": "mud", "relationship_type": "reversed"}
            print("  → 检测到反向关系：低PC值=砂岩，高PC值=泥岩")

        # 添加统计信息
        relationship.update(
            {
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "low_avg_thickness": low_pc_avg_thickness,
                "high_avg_thickness": high_pc_avg_thickness,
            }
        )

        return relationship

    def add_virtual_points_smart(
        self,
        mud_range=None,  # 手动指定泥岩区间 (start, end)
        sand_range=None,  # 手动指定砂岩区间 (start, end)
        n_points=20,
        noise_factor=0.1,
        auto_detect=True,
        primary_feature=None,
        placement_strategy="conservative",  # "conservative" 或 "extended"
    ):
        """
        智能添加虚拟点，支持手动设置和自动策略

        Parameters:
        -----------
        mud_range : tuple or None, default=None
            手动指定泥岩虚拟点范围 (start, end)
            例如: (-2, 0) 表示在PC1值-2到0之间添加泥岩虚拟点
        sand_range : tuple or None, default=None
            手动指定砂岩虚拟点范围 (start, end)
            例如: (2, 4) 表示在PC1值2到4之间添加砂岩虚拟点
        n_points : int, default=20
            每个区间生成的虚拟点数量
        noise_factor : float, default=0.1
            噪音因子，用于为虚拟点添加随机变化
        auto_detect : bool, default=True
            是否自动检测PC值与地质类型的关系
        primary_feature : str or None, default=None
            用于添加虚拟点的主要特征，如果为None则使用第一个特征
        placement_strategy : str, default="conservative"
            自动放置策略（仅在未手动指定范围时生效）:
            - "conservative": 在数据范围内侧保守放置（推荐）
            - "extended": 在数据范围外侧延伸放置

        Returns:
        --------
        tuple
            (enhanced_data, pc_geology_relationship)
            - enhanced_data: pd.DataFrame, 包含虚拟点的增强数据集
            - pc_geology_relationship: dict, PC值与地质类型的关系信息

        Examples:
        ---------
        # 手动指定泥岩区间
        data, relationship = model.add_virtual_points_smart(mud_range=(-5, 0), n_points=10)

        # 手动指定砂岩和泥岩区间
        data, relationship = model.add_virtual_points_smart(
            mud_range=(-2, 0), sand_range=(2, 5), n_points=10
        )

        # 使用保守的自动策略
        data, relationship = model.add_virtual_points_smart(
            placement_strategy="conservative", n_points=15
        )
        """

        if primary_feature is None:
            primary_feature = self.feature_columns[0]

        # 自动检测PC值与地质类型的关系
        if auto_detect:
            pc_geology_relationship = self.auto_detect_pc_geology_relationship(primary_feature)
        else:
            # 使用默认关系
            pc_geology_relationship = {"low_pc_type": "mud", "high_pc_type": "sand", "relationship_type": "default"}
            print(f"使用默认PC值关系：低PC值=泥岩，高PC值=砂岩")

        feature_min = self.data[primary_feature].min()
        feature_max = self.data[primary_feature].max()
        feature_range = feature_max - feature_min
        max_target = self.data[self.target_column].max()

        virtual_data = []

        print(f"虚拟点生成配置:")
        print(f"  主要特征: {primary_feature}")
        print(f"  数据范围: [{feature_min:.2f}, {feature_max:.2f}]")
        print(f"  每个区间点数: {n_points}")
        print(f"  噪音因子: {noise_factor}")

        # === 处理泥岩虚拟点 ===
        if mud_range is not None:
            # 手动指定泥岩区间
            print(f"  手动设置泥岩虚拟点范围: {mud_range}")
            mud_start, mud_end = mud_range

            # 生成泥岩虚拟点
            mud_x_values = np.linspace(mud_start, mud_end, n_points)
            for x_val in mud_x_values:
                virtual_point = {col: 0 for col in self.feature_columns}
                virtual_point[primary_feature] = x_val
                virtual_point[self.target_column] = abs(np.random.normal(0, noise_factor))
                virtual_point["is_virtual"] = True
                virtual_point["virtual_type"] = "mud"
                virtual_data.append(virtual_point)

        else:
            # 自动策略：根据PC-地质关系自动设置泥岩区间
            if placement_strategy == "conservative":
                # 保守策略：在数据范围内侧放置
                margin = feature_range * 0.15  # 15%的内缩边距

                if pc_geology_relationship["low_pc_type"] == "mud":
                    # 低PC值对应泥岩：在最小值右侧设置泥岩虚拟点
                    mud_start = feature_min
                    mud_end = feature_min + margin
                    print(f"  自动设置泥岩虚拟点（低PC=泥岩）: [{mud_start:.2f}, {mud_end:.2f}]")
                else:
                    # 高PC值对应泥岩：在最大值左侧设置泥岩虚拟点
                    mud_start = feature_max - margin
                    mud_end = feature_max
                    print(f"  自动设置泥岩虚拟点（高PC=泥岩）: [{mud_start:.2f}, {mud_end:.2f}]")

            else:  # extended strategy
                # 延伸策略：在数据范围外侧放置
                expansion = feature_range * 0.2  # 20%的外延

                if pc_geology_relationship["low_pc_type"] == "mud":
                    # 低PC值对应泥岩：在最小值左侧延伸
                    mud_start = feature_min - expansion
                    mud_end = feature_min
                    print(f"  自动设置泥岩虚拟点（低PC=泥岩，延伸）: [{mud_start:.2f}, {mud_end:.2f}]")
                else:
                    # 高PC值对应泥岩：在最大值右侧延伸
                    mud_start = feature_max
                    mud_end = feature_max + expansion
                    print(f"  自动设置泥岩虚拟点（高PC=泥岩，延伸）: [{mud_start:.2f}, {mud_end:.2f}]")

            # 生成泥岩虚拟点
            mud_x_values = np.linspace(mud_start, mud_end, n_points)
            for x_val in mud_x_values:
                virtual_point = {col: 0 for col in self.feature_columns}
                virtual_point[primary_feature] = x_val
                virtual_point[self.target_column] = abs(np.random.normal(0, noise_factor))
                virtual_point["is_virtual"] = True
                virtual_point["virtual_type"] = "mud"
                virtual_data.append(virtual_point)

        # === 处理砂岩虚拟点 ===
        if sand_range is not None:
            # 手动指定砂岩区间
            print(f"  手动设置砂岩虚拟点范围: {sand_range}")
            sand_start, sand_end = sand_range

            # 生成砂岩虚拟点
            sand_x_values = np.linspace(sand_start, sand_end, n_points)
            for x_val in sand_x_values:
                virtual_point = {col: 0 for col in self.feature_columns}
                virtual_point[primary_feature] = x_val
                virtual_point[self.target_column] = max_target + abs(np.random.normal(max_target * 0.1, noise_factor))
                virtual_point["is_virtual"] = True
                virtual_point["virtual_type"] = "sand"
                virtual_data.append(virtual_point)

        else:
            # 自动策略：根据PC-地质关系自动设置砂岩区间
            if placement_strategy == "conservative":
                # 保守策略：在数据范围内侧放置
                margin = feature_range * 0.15  # 15%的内缩边距

                if pc_geology_relationship["high_pc_type"] == "sand":
                    # 高PC值对应砂岩：在最大值左侧设置砂岩虚拟点
                    sand_start = feature_max - margin
                    sand_end = feature_max
                    print(f"  自动设置砂岩虚拟点（高PC=砂岩）: [{sand_start:.2f}, {sand_end:.2f}]")
                else:
                    # 低PC值对应砂岩：在最小值右侧设置砂岩虚拟点
                    sand_start = feature_min
                    sand_end = feature_min + margin
                    print(f"  自动设置砂岩虚拟点（低PC=砂岩）: [{sand_start:.2f}, {sand_end:.2f}]")

            else:  # extended strategy
                # 延伸策略：在数据范围外侧放置
                expansion = feature_range * 0.2  # 20%的外延

                if pc_geology_relationship["high_pc_type"] == "sand":
                    # 高PC值对应砂岩：在最大值右侧延伸
                    sand_start = feature_max
                    sand_end = feature_max + expansion
                    print(f"  自动设置砂岩虚拟点（高PC=砂岩，延伸）: [{sand_start:.2f}, {sand_end:.2f}]")
                else:
                    # 低PC值对应砂岩：在最小值左侧延伸
                    sand_start = feature_min - expansion
                    sand_end = feature_min
                    print(f"  自动设置砂岩虚拟点（低PC=砂岩，延伸）: [{sand_start:.2f}, {sand_end:.2f}]")

            # 生成砂岩虚拟点
            sand_x_values = np.linspace(sand_start, sand_end, n_points)
            for x_val in sand_x_values:
                virtual_point = {col: 0 for col in self.feature_columns}
                virtual_point[primary_feature] = x_val
                virtual_point[self.target_column] = max_target + abs(np.random.normal(max_target * 0.1, noise_factor))
                virtual_point["is_virtual"] = True
                virtual_point["virtual_type"] = "sand"
                virtual_data.append(virtual_point)

        # 合并数据
        enhanced_data = self.data.copy()
        enhanced_data["is_virtual"] = False
        enhanced_data["virtual_type"] = "real"

        if virtual_data:
            virtual_df = pd.DataFrame(virtual_data)
            enhanced_data = pd.concat([enhanced_data, virtual_df], ignore_index=True)
            print(f"  成功添加 {len(virtual_data)} 个虚拟点")

            # 统计虚拟点分布
            mud_count = sum(1 for vp in virtual_data if vp["virtual_type"] == "mud")
            sand_count = sum(1 for vp in virtual_data if vp["virtual_type"] == "sand")
            print(f"    - 泥岩虚拟点: {mud_count}")
            print(f"    - 砂岩虚拟点: {sand_count}")

        return enhanced_data, pc_geology_relationship

    def prepare_features(self, data, use_features=None, feature_weights=None):
        """
        准备特征，支持多维特征组合

        将多个PCA特征线性组合为单一输入特征，用于Sigmoid拟合。

        Parameters:
        -----------
        data : pd.DataFrame
            包含特征的数据源
        use_features : list or None, optional
            使用的特征列表，如['PC1', 'PC2']
            如果为None，则使用第一个特征
        feature_weights : list or None, optional
            特征权重列表，与use_features对应
            如果为None，则使用等权重

        Returns:
        --------
        np.array
            组合后的1D特征数组

        Notes:
        ------
        多特征组合公式：
        combined_feature = w1*PC1 + w2*PC2 + ... + wn*PCn
        其中 wi 为权重，通常基于PCA的方差贡献比设置
        """
        if use_features is None:
            use_features = [self.feature_columns[0]]

        if len(use_features) == 1:
            return data[use_features[0]].values

        # 多维特征线性组合
        if feature_weights is None:
            feature_weights = [1.0 / len(use_features)] * len(use_features)

        combined_features = np.zeros(len(data))
        for i, feature in enumerate(use_features):
            combined_features += feature_weights[i] * data[feature].values

        return combined_features

    def fit(
        self,
        use_features=None,
        feature_weights=None,
        virtual_points_config=None,
        bounds=None,
        initial_guess=None,
        max_iterations=2000,
    ):
        """
        拟合Sigmoid函数

        使用非线性最小二乘法拟合三参数Sigmoid函数到数据。

        Parameters:
        -----------
        use_features : list or None, optional
            使用的特征列表，如['PC1']或['PC1', 'PC2']
        feature_weights : list or None, optional
            特征权重，与use_features对应
        virtual_points_config : dict or None, optional
            虚拟点配置，支持两种模式：
            1. 智能模式: {'smart': True, 'n_points': int, 'noise_factor': float}
            2. 传统模式: {'x_mud': value, 'x_sand': value, 'n_points': int}
        bounds : tuple or None, optional
            参数边界 ((L_min, k_min, x0_min), (L_max, k_max, x0_max))
        initial_guess : tuple or None, optional
            初始参数猜测 (L, k, x0)
        max_iterations : int, default=2000
            优化算法最大迭代次数

        Returns:
        --------
        dict
            拟合结果字典，包含以下键：
            - 'success': bool, 拟合是否成功
            - 'params': dict, 拟合参数 {'L': float, 'k': float, 'x0': float}
            - 'param_errors': dict, 参数标准误差
            - 'r2_score': float, 决定系数
            - 'X': np.array, 输入特征
            - 'y': np.array, 目标值
            - 'y_pred': np.array, 预测值
            - 'use_features': list, 使用的特征
            - 'feature_weights': list, 特征权重
            如果失败，包含 'error': str

        Notes:
        ------
        拟合流程：
        1. 数据准备和虚拟点添加
        2. 特征组合和参数设置
        3. 非线性最小二乘拟合
        4. 结果评估和误差计算
        """
        # 准备数据
        working_data = self.data.copy()

        # 添加虚拟点
        if virtual_points_config:
            if virtual_points_config.get("smart", False):
                # 使用智能模式
                config = virtual_points_config.copy()
                config.pop("smart")  # 移除smart标志
                working_data, pc_relationship = self.add_virtual_points_smart(**config)
            else:
                # 使用传统模式
                working_data = self.add_virtual_points(**virtual_points_config)

        # 保存当前工作数据
        self.current_data = working_data

        # 准备特征
        X = self.prepare_features(working_data, use_features, feature_weights)
        y = working_data[self.target_column].values

        # 设置默认参数
        y_max = y.max()
        x_min, x_max = X.min(), X.max()
        x_range = x_max - x_min

        if bounds is None:
            bounds = (
                [y_max * 0.5, -10, x_min - x_range],  # 下界
                [y_max * 2.0, 10, x_max + x_range],  # 上界
            )

        if initial_guess is None:
            initial_guess = [y_max, 1.0, np.median(X)]

        try:
            # 拟合sigmoid函数
            self.fit_params, covariance = curve_fit(
                self.sigmoid, X, y, p0=initial_guess, bounds=bounds, maxfev=max_iterations
            )

            # 计算拟合质量
            y_pred = self.sigmoid(X, *self.fit_params)
            self.r2_score = r2_score(y, y_pred)

            # 计算参数标准误差
            param_errors = np.sqrt(np.diag(covariance))

            return {
                "success": True,
                "params": dict(zip(["L", "k", "x0"], self.fit_params)),
                "param_errors": dict(zip(["L_err", "k_err", "x0_err"], param_errors)),
                "r2_score": self.r2_score,
                "X": X,
                "y": y,
                "y_pred": y_pred,
                "use_features": use_features or [self.feature_columns[0]],
                "feature_weights": feature_weights,
            }

        except Exception as e:
            return {"success": False, "error": str(e), "X": X, "y": y}

    def predict(self, new_data, use_features=None, feature_weights=None):
        """
        使用拟合的模型进行预测

        对新数据应用已拟合的Sigmoid模型进行砂厚预测。

        Parameters:
        -----------
        new_data : pd.DataFrame or np.array
            新的输入数据
            - 如果是DataFrame，必须包含use_features中指定的列
            - 如果是numpy数组，视为已处理的1D特征
        use_features : list or None, optional
            使用的特征列表，应与拟合时一致
        feature_weights : list or None, optional
            特征权重，应与拟合时一致

        Returns:
        --------
        np.array
            预测的砂厚值数组

        Raises:
        -------
        ValueError
            当模型尚未拟合时抛出异常

        Notes:
        ------
        预测流程：
        1. 检查模型是否已拟合
        2. 特征准备和组合
        3. 应用Sigmoid函数
        """
        if self.fit_params is None:
            raise ValueError("模型尚未拟合，请先调用fit方法")

        if isinstance(new_data, pd.DataFrame):
            X_new = self.prepare_features(new_data, use_features, feature_weights)
        else:
            X_new = new_data

        return self.sigmoid(X_new, *self.fit_params)

    def visualize_fit(self, fit_result, figsize=(15, 8), save_path=None):
        """
        可视化拟合结果

        生成包含拟合曲线、残差分析和模型信息的综合可视化图表。

        Parameters:
        -----------
        fit_result : dict
            fit方法返回的拟合结果字典
        figsize : tuple, default=(15, 8)
            图形大小 (width, height)
        save_path : str or None, optional
            图片保存路径，如果为None则不保存

        Returns:
        --------
        matplotlib.figure.Figure or None
            生成的图形对象，如果拟合失败则返回None

        Notes:
        ------
        可视化内容：
        1. 左图：散点图 + Sigmoid拟合曲线 + 虚拟点标识
        2. 右图：残差分析图
        3. 模型参数和质量指标文本框
        """
        if not fit_result["success"]:
            print(f"拟合失败: {fit_result['error']}")
            return None

        # 提取数据
        X = fit_result["X"]
        y = fit_result["y"]
        y_pred = fit_result["y_pred"]
        params = fit_result["params"]
        r2_score_val = fit_result["r2_score"]

        # 创建图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 左图：拟合结果
        if self.current_data is not None and "is_virtual" in self.current_data.columns:
            # 区分真实点和虚拟点
            real_mask = ~self.current_data["is_virtual"]
            virtual_mask = self.current_data["is_virtual"]

            # 真实点
            ax1.scatter(
                X[real_mask],
                y[real_mask],
                c="blue",
                alpha=0.7,
                s=60,
                label="真实样本",
                edgecolors="black",
                linewidth=0.5,
            )

            # 虚拟点
            if virtual_mask.any():
                mud_mask = self.current_data["virtual_type"] == "mud"
                sand_mask = self.current_data["virtual_type"] == "sand"

                if mud_mask.any():
                    ax1.scatter(X[mud_mask], y[mud_mask], c="brown", alpha=0.5, s=30, marker="^", label="虚拟点(泥岩)")
                if sand_mask.any():
                    ax1.scatter(
                        X[sand_mask], y[sand_mask], c="orange", alpha=0.5, s=30, marker="v", label="虚拟点(砂岩)"
                    )
        else:
            ax1.scatter(X, y, c="blue", alpha=0.7, s=60, label="样本点", edgecolors="black", linewidth=0.5)

        # 绘制拟合曲线
        X_curve = np.linspace(X.min(), X.max(), 300)
        y_curve = self.sigmoid(X_curve, *self.fit_params)
        ax1.plot(X_curve, y_curve, "red", linewidth=2, label="Sigmoid拟合")

        # 添加模型信息
        param_text = f"L = {params['L']:.2f}\n"
        param_text += f"k = {params['k']:.3f}\n"
        param_text += f"x_0 = {params['x0']:.2f}\n"
        param_text += f"R^2 = {r2_score_val:.3f}"

        ax1.text(
            0.02,
            0.98,
            param_text,
            transform=ax1.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        ax1.set_xlabel("特征值")
        ax1.set_ylabel("砂厚 (m)")
        ax1.set_title("Sigmoid函数拟合结果")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 右图：残差分析
        residuals = y - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, c="green", s=40)
        ax2.axhline(y=0, color="red", linestyle="--", alpha=0.8)
        ax2.set_xlabel("预测值 (m)")
        ax2.set_ylabel("残差 (m)")
        ax2.set_title("残差分析")
        ax2.grid(True, alpha=0.3)

        # 添加残差统计
        residual_stats = f"残差均值: {np.mean(residuals):.3f}\n"
        residual_stats += f"残差标准差: {np.std(residuals):.3f}\n"
        residual_stats += f"RMSE: {np.sqrt(np.mean(residuals**2)):.3f}"

        ax2.text(
            0.02,
            0.98,
            residual_stats,
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig
