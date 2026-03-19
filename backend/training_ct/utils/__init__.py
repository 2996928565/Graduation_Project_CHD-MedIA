"""
CT 训练工具包
"""

from .ct_preprocessing import (
    apply_ct_window,
    normalize_hu_values,
    resample_volume,
    ct_preprocess_pipeline,
    CT_WINDOW_PRESETS,
)

from .label_utils import (
    auto_detect_labels,
    remap_labels,
    validate_label_mapping,
    get_label_distribution,
)

from .visualization import (
    visualize_slice,
    visualize_3d_volume,
    save_prediction_comparison,
)

__all__ = [
    # CT 预处理
    'apply_ct_window',
    'normalize_hu_values',
    'resample_volume',
    'ct_preprocess_pipeline',
    'CT_WINDOW_PRESETS',

    # 标签工具
    'auto_detect_labels',
    'remap_labels',
    'validate_label_mapping',
    'get_label_distribution',

    # 可视化
    'visualize_slice',
    'visualize_3d_volume',
    'save_prediction_comparison',
]
