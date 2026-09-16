"""模型层统一异常：真实数据不足以建模时显式失败，由编排层转为 unavailable，
而不是用残缺/合成数据硬算出一个看似精确的预测。"""


class InsufficientData(RuntimeError):
    """真实观测样本不足或关键字段缺失，无法训练该模型。"""
