import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import t
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
        """
        self.data = data.copy()
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.fit_params = None
        self.covariance_matrix = None  # 新增：存储协方差矩阵
        self.residual_std = None  # 新增：存储残差标准差
        self.r2_score = None
        self.current_data = None

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
            如果指定，则只添加泥岩虚拟点，不自动添加砂岩虚拟点
        sand_range : tuple or None, default=None
            手动指定砂岩虚拟点范围 (start, end)
            例如: (2, 4) 表示在PC1值2到4之间添加砂岩虚拟点
            如果指定，则只添加砂岩虚拟点，不自动添加泥岩虚拟点
        n_points : int, default=20
            每个区间生成的虚拟点数量
        noise_factor : float, default=0.1
            噪音因子，用于为虚拟点添加随机变化
        auto_detect : bool, default=True
            是否自动检测PC值与地质类型的关系（仅在全自动模式下使用）
        primary_feature : str or None, default=None
            用于添加虚拟点的主要特征，如果为None则使用第一个特征
        placement_strategy : str, default="conservative"
            自动放置策略（仅在全自动模式下使用）:
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
        # 只设置泥岩虚拟点
        data, relationship = model.add_virtual_points_smart(mud_range=(-5, 0), n_points=10)

        # 只设置砂岩虚拟点
        data, relationship = model.add_virtual_points_smart(sand_range=(2, 5), n_points=10)

        # 手动设置砂岩和泥岩区间
        data, relationship = model.add_virtual_points_smart(
            mud_range=(-2, 0), sand_range=(2, 5), n_points=10
        )

        # 全自动模式（两种都会设置）
        data, relationship = model.add_virtual_points_smart(
            placement_strategy="conservative", n_points=15
        )
        """

        if primary_feature is None:
            primary_feature = self.feature_columns[0]

        feature_min = self.data[primary_feature].min()
        feature_max = self.data[primary_feature].max()
        feature_range = feature_max - feature_min
        max_target = self.data[self.target_column].max()

        virtual_data = []
        pc_geology_relationship = None

        print(f"虚拟点生成配置:")
        print(f"  主要特征: {primary_feature}")
        print(f"  数据范围: [{feature_min:.2f}, {feature_max:.2f}]")
        print(f"  每个区间点数: {n_points}")
        print(f"  噪音因子: {noise_factor}")

        # 判断是否为全自动模式（两个区间都未手动指定）
        is_full_auto = (mud_range is None) and (sand_range is None)

        if is_full_auto:
            # 全自动模式：自动检测关系并设置两种虚拟点
            if auto_detect:
                pc_geology_relationship = self.auto_detect_pc_geology_relationship(primary_feature)
            else:
                pc_geology_relationship = {"low_pc_type": "mud", "high_pc_type": "sand", "relationship_type": "default"}
                print(f"使用默认PC值关系：低PC值=泥岩，高PC值=砂岩")

            print(f"  模式: 全自动模式")
            print(f"  放置策略: {placement_strategy}")

            # 自动设置泥岩虚拟点
            if placement_strategy == "conservative":
                margin = feature_range * 0.15
                if pc_geology_relationship["low_pc_type"] == "mud":
                    mud_start, mud_end = feature_min, feature_min + margin
                    print(f"  自动设置泥岩虚拟点（低PC=泥岩）: [{mud_start:.2f}, {mud_end:.2f}]")
                else:
                    mud_start, mud_end = feature_max - margin, feature_max
                    print(f"  自动设置泥岩虚拟点（高PC=泥岩）: [{mud_start:.2f}, {mud_end:.2f}]")
            else:  # extended
                expansion = feature_range * 0.2
                if pc_geology_relationship["low_pc_type"] == "mud":
                    mud_start, mud_end = feature_min - expansion, feature_min
                    print(f"  自动设置泥岩虚拟点（低PC=泥岩，延伸）: [{mud_start:.2f}, {mud_end:.2f}]")
                else:
                    mud_start, mud_end = feature_max, feature_max + expansion
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

            # 自动设置砂岩虚拟点
            if placement_strategy == "conservative":
                margin = feature_range * 0.15
                if pc_geology_relationship["high_pc_type"] == "sand":
                    sand_start, sand_end = feature_max - margin, feature_max
                    print(f"  自动设置砂岩虚拟点（高PC=砂岩）: [{sand_start:.2f}, {sand_end:.2f}]")
                else:
                    sand_start, sand_end = feature_min, feature_min + margin
                    print(f"  自动设置砂岩虚拟点（低PC=砂岩）: [{sand_start:.2f}, {sand_end:.2f}]")
            else:  # extended
                expansion = feature_range * 0.2
                if pc_geology_relationship["high_pc_type"] == "sand":
                    sand_start, sand_end = feature_max, feature_max + expansion
                    print(f"  自动设置砂岩虚拟点（高PC=砂岩，延伸）: [{sand_start:.2f}, {sand_end:.2f}]")
                else:
                    sand_start, sand_end = feature_min - expansion, feature_min
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

        else:
            # 手动模式：只设置指定的虚拟点类型
            print(f"  模式: 手动模式")

            # 设置一个简单的关系信息用于返回
            pc_geology_relationship = {"relationship_type": "manual"}

            # === 处理泥岩虚拟点（仅在手动指定时） ===
            if mud_range is not None:
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

            # === 处理砂岩虚拟点（仅在手动指定时） ===
            if sand_range is not None:
                print(f"  手动设置砂岩虚拟点范围: {sand_range}")
                sand_start, sand_end = sand_range

                # 生成砂岩虚拟点
                sand_x_values = np.linspace(sand_start, sand_end, n_points)
                for x_val in sand_x_values:
                    virtual_point = {col: 0 for col in self.feature_columns}
                    virtual_point[primary_feature] = x_val
                    virtual_point[self.target_column] = max_target + abs(
                        np.random.normal(max_target * 0.1, noise_factor)
                    )
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
        else:
            print(f"  未添加任何虚拟点")

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
            虚拟点配置
        bounds : tuple or None, optional
            参数边界 ((L_min, k_min, x0_min), (L_max, k_max, x0_max))
        initial_guess : tuple or None, optional
            初始参数猜测 (L, k, x0)
        max_iterations : int, default=2000
            优化算法最大迭代次数

        Returns:
        --------
        dict
            拟合结果字典，包含置信带相关参数
        """
        # 准备数据
        working_data = self.data.copy()

        # 添加虚拟点
        if virtual_points_config:
            config = virtual_points_config.copy()
            config.pop("smart", None)
            working_data, pc_relationship = self.add_virtual_points_smart(**config)

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
                [y_max * 0.5, -10, x_min - x_range],
                [y_max * 2.0, 10, x_max + x_range],
            )

        if initial_guess is None:
            initial_guess = [y_max, 1.0, np.median(X)]

        try:
            # 拟合sigmoid函数
            self.fit_params, self.covariance_matrix = curve_fit(
                self.sigmoid, X, y, p0=initial_guess, bounds=bounds, maxfev=max_iterations
            )

            # 计算拟合质量
            y_pred = self.sigmoid(X, *self.fit_params)
            self.r2_score = r2_score(y, y_pred)

            # 计算残差标准差
            residuals = y - y_pred
            self.residual_std = np.std(residuals, ddof=len(self.fit_params))

            # 计算参数标准误差
            param_errors = np.sqrt(np.diag(self.covariance_matrix))

            return {
                "success": True,
                "params": dict(zip(["L", "k", "x0"], self.fit_params)),
                "param_errors": dict(zip(["L_err", "k_err", "x0_err"], param_errors)),
                "r2_score": self.r2_score,
                "covariance_matrix": self.covariance_matrix,  # 新增
                "residual_std": self.residual_std,  # 新增
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

    def get_confidence_band(self, x_values, confidence_level=0.95):
        """
        计算给定x值的Sigmoid曲线的置信带

        Parameters:
        -----------
        x_values : np.array
            需要计算置信带的x值数组
        confidence_level : float, default=0.95
            置信水平 (例如 0.95 表示 95%)

        Returns:
        --------
        tuple
            (lower_bound, upper_bound, margin_of_error)
            包含置信带下界、上界和误差幅度的元组

        Raises:
        -------
        ValueError
            当模型尚未拟合或缺少协方差矩阵时抛出异常
        """
        if self.fit_params is None or self.covariance_matrix is None:
            raise ValueError("模型尚未拟合或缺少协方差矩阵，请先调用fit方法")

        L, k, x0 = self.fit_params
        n = len(self.current_data)  # 样本数量
        p = len(self.fit_params)  # 参数数量
        dof = max(1, n - p)  # 自由度

        # t分布的临界值
        alpha = 1.0 - confidence_level
        t_val = t.ppf(1.0 - alpha / 2.0, dof)

        lower_bound, upper_bound, margin_errors = [], [], []

        for x in x_values:
            # 计算雅可比矩阵 (Sigmoid函数对各参数的偏导数)
            exp_term = np.exp(-k * (x - x0))
            denom = 1.0 + exp_term

            # 对L的偏导数
            dL = 1.0 / denom

            # 对k的偏导数
            dk = L * (x - x0) * exp_term / (denom**2)

            # 对x0的偏导数
            dx0 = L * k * exp_term / (denom**2)

            jacobian = np.array([dL, dk, dx0])

            # 计算标准误差: se = sqrt(J * C * J^T)
            se = np.sqrt(np.dot(jacobian, np.dot(self.covariance_matrix, jacobian.T)))

            # 计算置信区间
            margin_of_error = t_val * se
            y_pred = self.sigmoid(x, *self.fit_params)

            lower_bound.append(y_pred - margin_of_error)
            upper_bound.append(y_pred + margin_of_error)
            margin_errors.append(margin_of_error)

        return np.array(lower_bound), np.array(upper_bound), np.array(margin_errors)

    def get_prediction_interval(self, x_values, confidence_level=0.95):
        """
        计算预测区间（比置信带更宽，包含数据噪声的不确定性）

        Parameters:
        -----------
        x_values : np.array
            需要计算预测区间的x值数组
        confidence_level : float, default=0.95
            置信水平

        Returns:
        --------
        tuple
            (lower_bound, upper_bound, margin_of_error)
            包含预测区间下界、上界和误差幅度的元组
        """
        if self.residual_std is None:
            raise ValueError("模型尚未拟合或缺少残差标准差信息")

        # 先获取置信带
        conf_lower, conf_upper, conf_margin = self.get_confidence_band(x_values, confidence_level)

        # 添加残差标准差的贡献
        n = len(self.current_data)
        p = len(self.fit_params)
        dof = max(1, n - p)
        alpha = 1.0 - confidence_level
        t_val = t.ppf(1.0 - alpha / 2.0, dof)

        # 预测区间的额外不确定性
        additional_error = t_val * self.residual_std

        pred_lower = conf_lower - additional_error
        pred_upper = conf_upper + additional_error
        pred_margin = conf_margin + additional_error

        return pred_lower, pred_upper, pred_margin

    def predict_with_uncertainty(self, new_data, use_features=None, feature_weights=None, confidence_level=0.95):
        """
        带不确定性的预测

        Parameters:
        -----------
        new_data : pd.DataFrame or np.array
            新的输入数据
        use_features : list or None, optional
            使用的特征列表
        feature_weights : list or None, optional
            特征权重
        confidence_level : float, default=0.95
            置信水平

        Returns:
        --------
        dict
            包含预测值、置信带和预测区间的字典
        """
        if self.fit_params is None:
            raise ValueError("模型尚未拟合，请先调用fit方法")

        if isinstance(new_data, pd.DataFrame):
            X_new = self.prepare_features(new_data, use_features, feature_weights)
        else:
            X_new = new_data

        # 基本预测
        y_pred = self.sigmoid(X_new, *self.fit_params)

        # 置信带
        conf_lower, conf_upper, conf_margin = self.get_confidence_band(X_new, confidence_level)

        # 预测区间
        pred_lower, pred_upper, pred_margin = self.get_prediction_interval(X_new, confidence_level)

        return {
            "predictions": y_pred,
            "confidence_lower": conf_lower,
            "confidence_upper": conf_upper,
            "confidence_margin": conf_margin,
            "prediction_lower": pred_lower,
            "prediction_upper": pred_upper,
            "prediction_margin": pred_margin,
            "confidence_level": confidence_level,
        }

    def visualize_predict(
        self,
        true_values,
        predicted_values,
        feature_values=None,
        use_features=None,
        feature_weights=None,
        show_confidence_band=True,
        confidence_level=0.95,
        figsize=(15, 6),
        save_path=None,
    ):
        """
        可视化预测结果对比

        专门用于预测模式的可视化，右图显示置信区间。

        Parameters:
        -----------
        true_values : array-like
            真实值数组
        predicted_values : array-like
            预测值数组
        feature_values : array-like, optional
            特征值数组（用于计算置信区间）
        use_features : list, optional
            使用的特征列表
        feature_weights : list, optional
            特征权重
        show_confidence_band : bool, default=True
            是否在右图显示置信区间
        confidence_level : float, default=0.95
            置信水平
        figsize : tuple, default=(15, 6)
            图形大小
        save_path : str, optional
            保存路径

        Returns:
        --------
        matplotlib.figure.Figure
            生成的图形对象
        """
        if self.fit_params is None:
            print("WARNING: 模型尚未拟合，无法显示置信带")
            show_confidence_band = False

        # 转换为numpy数组
        true_values = np.array(true_values)
        predicted_values = np.array(predicted_values)

        if feature_values is not None:
            feature_values = np.array(feature_values)

        # 创建图像
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 左图：预测 vs 真实值散点图（简洁版，不显示置信带）
        ax1.scatter(true_values, predicted_values, alpha=0.7, s=80, edgecolors="black", label="井点预测")

        # 添加1:1参考线
        min_val = min(true_values.min(), predicted_values.min())
        max_val = max(true_values.max(), predicted_values.max())
        ax1.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, linewidth=2, label="1:1参考线")

        ax1.set_xlabel("真实砂厚 (m)")
        ax1.set_ylabel("预测砂厚 (m)")
        ax1.set_title("预测 vs 真实砂厚")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 右图：残差散点图（带置信区间）
        residuals = true_values - predicted_values
        ax2.scatter(predicted_values, residuals, alpha=0.7, s=60, edgecolors="black", label="残差点")
        ax2.axhline(y=0, color="red", linestyle="--", linewidth=2, label="零残差线")

        # 在右图显示置信区间
        confidence_info = ""
        if show_confidence_band and feature_values is not None:
            try:
                # 计算置信带
                conf_lower, conf_upper, conf_margin = self.get_confidence_band(feature_values, confidence_level)

                # 在残差图中显示置信区间
                residual_ci = np.mean(conf_margin)
                ax2.axhline(
                    y=residual_ci,
                    color="orange",
                    linestyle=":",
                    alpha=0.7,
                    linewidth=2,
                    label=f"±{confidence_level * 100:.0f}%置信区间",
                )
                ax2.axhline(y=-residual_ci, color="orange", linestyle=":", alpha=0.7, linewidth=2)

                # 填充置信区间
                xlim = ax2.get_xlim()
                ax2.fill_between(xlim, -residual_ci, residual_ci, alpha=0.1, color="orange")
                ax2.set_xlim(xlim)  # 恢复x轴范围

                # 计算残差在置信区间内的比例
                within_ci = np.abs(residuals) <= residual_ci
                within_ci_ratio = np.mean(within_ci) * 100

                confidence_info = f"\n置信区间分析 ({confidence_level * 100:.0f}%):\n"
                confidence_info += f"  平均置信区间: ±{residual_ci:.3f} m\n"
                confidence_info += f"  残差在置信区间内: {within_ci_ratio:.1f}%"

            except Exception as e:
                print(f"WARNING: 置信区间计算失败: {e}")
                confidence_info = "\n置信区间计算失败"

        ax2.set_xlabel("预测值 (m)")
        ax2.set_ylabel("残差 (真实 - 预测) (m)")
        ax2.set_title("残差分析")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        # 计算并打印统计信息
        correlation = np.corrcoef(true_values, predicted_values)[0, 1]
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))

        print(f"\n=== 预测结果摘要 ===")
        print(f"预测性能:")
        print(f"  预测-实际相关系数: {correlation:.3f}")
        print(f"  RMSE: {rmse:.3f} m")
        print(f"  MAE: {mae:.3f} m")
        print(f"  残差统计: 均值={np.mean(residuals):.3f}, 标准差={np.std(residuals):.3f}")
        print(confidence_info)
        print(f"数据统计:")
        print(f"  井点砂厚范围: {true_values.min():.2f} - {true_values.max():.2f} m")
        print(f"  预测砂厚范围: {predicted_values.min():.2f} - {predicted_values.max():.2f} m")
        print(f"  井点数量: {len(true_values)}")

        return fig
