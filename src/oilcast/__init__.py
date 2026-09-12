"""OilCast —— 多因素油价智能分析与预测系统。

模块划分：
    data_sources  数据获取（真实 API/RSS + 合成数据兜底）
    storage       SQLite / CSV 持久化
    features      特征工程
    models        权重学习、短/中/长期预测模型
    reporting     静态 HTML 报告与文字解读
    pipeline      每日主流程
    app           Streamlit 交互页面
"""

__version__ = "1.0.0"
