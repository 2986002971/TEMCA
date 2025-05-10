"""独立实现的Vietoris-Rips复形计算模块"""

from itertools import starmap

import numpy as np
import torch
from gph import ripser_parallel  # 只需安装giotto-ph
from torch import nn


class PersistenceInformation:
    """存储持久同调信息的数据结构"""

    def __init__(self, pairing, diagram, dimension):
        """
        初始化持久信息对象

        参数:
        ------
        pairing : numpy.ndarray
            持久配对，表示哪些单纯形创建/销毁了拓扑特征

        diagram : torch.Tensor
            持久图，包含(出生,死亡)对

        dimension : int
            拓扑维度
        """
        self.pairing = pairing
        self.diagram = diagram
        self.dimension = dimension

    def __repr__(self):
        return f"PersistenceInformation(dimension={self.dimension}, features={len(self.diagram)})"


def batch_handler(x, fn, **kwargs):
    """处理批量输入的函数

    参数:
    ------
    x : torch.Tensor 或 list
        输入数据，可以是张量或列表

    fn : callable
        要应用于每个样本的函数

    kwargs : dict
        传递给fn的额外参数

    返回:
    -------
    list 或 list of lists
        函数应用结果
    """
    # 处理列表输入
    if isinstance(x, list):
        return [fn(x_i, **kwargs) for x_i in x]

    # 如果是单个点云 (2D张量)
    if len(x.shape) == 2:
        return fn(x, **kwargs)

    # 如果是批量点云 (3D张量)
    return [fn(x_i, **kwargs) for x_i in x]


class VietorisRipsComplex(nn.Module):
    """计算点云数据的Vietoris-Rips复形和持久同调"""

    def __init__(
        self, dim=1, p=2, threshold=float("inf"), keep_infinite_features=False, **kwargs
    ):
        """
        初始化模块

        参数:
        ------
        dim : int
            计算持久同调的最高维度

        p : float
            计算点之间距离使用的Minkowski p-范数

        threshold : float
            拓扑特征计算的距离阈值

        keep_infinite_features : bool
            是否保留无限持久的特征

        **kwargs :
            传递给ripser的其他参数
        """
        super().__init__()

        self.dim = dim
        self.p = p
        self.threshold = threshold
        self.keep_infinite_features = keep_infinite_features

        # 确保每次调用ripser时使用相同的参数
        self.ripser_params = {
            "return_generators": True,
            "maxdim": self.dim,
            "thresh": self.threshold,
        }
        self.ripser_params.update(kwargs)

    def forward(self, x, treat_as_distances=False):
        """
        前向传播计算持久图

        参数:
        ------
        x : torch.Tensor 或 list
            输入点云，可以是形状为(n,d)的单个点云
            或形状为(b,n,d)的批量点云，或点云列表

        treat_as_distances : bool
            是否将x视为预先计算的距离矩阵

        返回:
        -------
        list of PersistenceInformation
            包含持久图和生成元的持久信息列表
        """
        return batch_handler(x, self._forward, treat_as_distances=treat_as_distances)

    def _forward(self, x, treat_as_distances=False):
        """处理单个点云的内部函数"""
        if treat_as_distances:
            distances = x
        else:
            distances = torch.cdist(x, x, p=self.p)

        # 调用ripser计算持久同调
        generators = ripser_parallel(
            distances.cpu().detach().numpy(), metric="precomputed", **self.ripser_params
        )["gens"]

        # 首先处理0维信息
        persistence_information = self._extract_generators_and_diagrams(
            distances,
            generators,
            dim0=True,
        )

        if self.keep_infinite_features:
            persistence_information_inf = self._extract_generators_and_diagrams(
                distances,
                generators,
                finite=False,
                dim0=True,
            )

        # 检查是否有更高维度的信息
        if self.dim >= 1:
            persistence_information.extend(
                self._extract_generators_and_diagrams(
                    distances,
                    generators,
                    dim0=False,
                )
            )

            if self.keep_infinite_features:
                persistence_information_inf.extend(
                    self._extract_generators_and_diagrams(
                        distances,
                        generators,
                        finite=False,
                        dim0=False,
                    )
                )

        # 合并有限和无限特征
        if self.keep_infinite_features:
            persistence_information = self._concatenate_features(
                persistence_information, persistence_information_inf
            )

        return persistence_information

    def _extract_generators_and_diagrams(self, dist, gens, finite=True, dim0=False):
        """从原始数据中提取生成元和持久图"""
        index = 1 if not dim0 else 0

        # 索引偏移以查找无限特征
        if not finite:
            index += 2

        gens = gens[index]

        if dim0:
            if finite:
                # Vietoris-Rips复形中所有顶点在时间0创建
                creators = torch.zeros_like(
                    torch.as_tensor(gens)[:, 0], device=dist.device
                )

                destroyers = dist[gens[:, 1], gens[:, 2]]
            else:
                creators = torch.zeros_like(
                    torch.as_tensor(gens)[:], device=dist.device
                )

                destroyers = torch.full_like(
                    torch.as_tensor(gens)[:],
                    torch.inf,
                    dtype=torch.float,
                    device=dist.device,
                )

                inf_pairs = np.full(shape=(gens.shape[0], 2), fill_value=-1)
                gens = np.column_stack((gens, inf_pairs))

            persistence_diagram = torch.stack((creators, destroyers), 1)

            return [PersistenceInformation(gens, persistence_diagram, 0)]
        else:
            result = []

            for index, gens_ in enumerate(gens):
                # 维度0特殊处理，所以这里需要偏移
                dimension = index + 1

                if finite:
                    creators = dist[gens_[:, 0], gens_[:, 1]]
                    destroyers = dist[gens_[:, 2], gens_[:, 3]]

                    persistence_diagram = torch.stack((creators, destroyers), 1)
                else:
                    creators = dist[gens_[:, 0], gens_[:, 1]]

                    destroyers = torch.full_like(
                        torch.as_tensor(gens_)[:, 0],
                        torch.inf,
                        dtype=torch.float,
                        device=dist.device,
                    )

                    # 创建特殊的无限对
                    inf_pairs = np.full(shape=(gens_.shape[0], 2), fill_value=-1)
                    gens_ = np.column_stack((gens_, inf_pairs))

                persistence_diagram = torch.stack((creators, destroyers), 1)

                result.append(
                    PersistenceInformation(gens_, persistence_diagram, dimension)
                )

        return result

    def _concatenate_features(self, pers_info_finite, pers_info_infinite):
        """连接有限和无限特征"""

        def _apply(fin, inf):
            assert fin.dimension == inf.dimension

            diagram = torch.concat((fin.diagram, inf.diagram))
            pairing = np.concatenate((fin.pairing, inf.pairing), axis=0)
            dimension = fin.dimension

            return PersistenceInformation(
                pairing=pairing, diagram=diagram, dimension=dimension
            )

        return list(starmap(_apply, zip(pers_info_finite, pers_info_infinite)))


# 使用示例
if __name__ == "__main__":
    # 创建一个简单的环形点云
    n_points = 30
    t = torch.linspace(0, 2 * np.pi, n_points)
    circle = torch.stack([torch.cos(t), torch.sin(t)], dim=1)

    # 添加一点随机噪声
    circle = circle + torch.randn_like(circle) * 0.05

    # 初始化VR复形
    vr = VietorisRipsComplex(dim=1, threshold=2.0)

    # 计算持久同调
    persistence_info = vr(circle)

    # 打印结果
    for pi in persistence_info:
        print(f"维度 {pi.dimension}: 特征数量 {len(pi.diagram)}")
        print("持久图:\n", pi.diagram)
