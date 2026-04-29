"""
MRI 模型推理脚本（无需标签）
仅进行预测，保存分割结果
"""
import sys
import json
from pathlib import Path
import argparse

import torch
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.model import get_model
from training.dataset import normalize_intensity, MMWHS_CLASS_NAMES
from training.normal_heart_model import extract_case_features, load_normal_heart_model


def predict_full_volume_sliding(
    model,
    image: np.ndarray,
    device,
    patch_size=(64, 128, 128),
    stride=(32, 64, 64),
):
    """滑窗预测完整3D体积，避免整幅前向导致显存溢出"""
    model.eval()
    d, h, w = image.shape
    pd, ph, pw = patch_size
    sd, sh, sw = stride

    num_classes = 8
    prob_sum = np.zeros((num_classes, d, h, w), dtype=np.float32)
    count_map = np.zeros((d, h, w), dtype=np.float32)

    def _starts(size, patch, step):
        if size <= patch:
            return [0]
        starts = list(range(0, size - patch + 1, step))
        tail = size - patch
        if starts[-1] != tail:
            starts.append(tail)
        return starts

    d_starts = _starts(d, pd, sd)
    h_starts = _starts(h, ph, sh)
    w_starts = _starts(w, pw, sw)

    total_patches = len(d_starts) * len(h_starts) * len(w_starts)
    with torch.no_grad():
        with tqdm(total=total_patches, desc="Predicting patches") as pbar:
            for ds in d_starts:
                de = min(ds + pd, d)
                ds = de - pd
                for hs in h_starts:
                    he = min(hs + ph, h)
                    hs = he - ph
                    for ws in w_starts:
                        we = min(ws + pw, w)
                        ws = we - pw

                        patch = image[ds:de, hs:he, ws:we]
                        patch_t = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(device)
                        logits = model(patch_t)
                        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                        prob_sum[:, ds:de, hs:he, ws:we] += probs
                        count_map[ds:de, hs:he, ws:we] += 1.0
                        pbar.update(1)

    count_map = np.maximum(count_map, 1.0)
    prob_avg = prob_sum / count_map[np.newaxis, ...]
    prediction = np.argmax(prob_avg, axis=0).astype(np.uint8)
    return prediction


def predict_single_image(
    model,
    image_path,
    device,
    output_dir,
    normal_model=None,
    labels_only=False,
    patch_size=(64, 128, 128),
    stride=(32, 64, 64),
):
    """预测单个图像（无需标签）"""
    model.eval()
    
    # 加载数据
    case_name = Path(image_path).stem.replace("_image", "")
    print(f"\n处理: {case_name}")
    
    img_sitk = sitk.ReadImage(str(image_path))
    img_arr = sitk.GetArrayFromImage(img_sitk).astype(np.float32)
    
    print(f"图像尺寸: {img_arr.shape}")
    
    # 预处理
    img_arr = normalize_intensity(img_arr)
    
    # 转为tensor
    img_tensor = torch.from_numpy(img_arr).unsqueeze(0).unsqueeze(0).float().to(device)
    
    print("开始预测（滑窗3D）...")
    prediction = predict_full_volume_sliding(
        model=model,
        image=img_arr,
        device=device,
        patch_size=tuple(patch_size),
        stride=tuple(stride),
    )
    
    # 统计预测结果
    print("\n预测统计:")
    unique, counts = np.unique(prediction, return_counts=True)
    total_voxels = prediction.size
    
    for cls_idx, count in zip(unique, counts):
        cls_name = MMWHS_CLASS_NAMES[cls_idx] if cls_idx < len(MMWHS_CLASS_NAMES) else f"Unknown-{cls_idx}"
        percentage = (count / total_voxels) * 100
        print(f"  {cls_name:<20s}: {count:>10d} voxels ({percentage:>5.2f}%)")
    
    # 保存预测结果
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pred_sitk = sitk.GetImageFromArray(prediction)
    pred_sitk.CopyInformation(img_sitk)
    pred_path = output_dir / f"{case_name}_prediction.nii.gz"
    sitk.WriteImage(pred_sitk, str(pred_path))
    print(f"\n✓ 预测结果已保存: {pred_path}")

    normality_result = None
    if (normal_model is not None) and (not labels_only):
        spacing_xyz = tuple(float(v) for v in img_sitk.GetSpacing())
        features = extract_case_features(prediction, spacing_xyz=spacing_xyz)
        normality_result = normal_model.evaluate_features(features)
        status = "异常" if normality_result["is_abnormal"] else "正常"
        print(
            f"常模判别: {status} | "
            f"score={normality_result['score']:.4f} "
            f"(thr={normality_result['score_threshold']:.4f})"
        )
        if normality_result["abnormal_features"]:
            print("  异常特征Top3:")
            for item in normality_result["abnormal_features"][:3]:
                print(f"    - {item['feature']}: z={item['z_score']:.3f}, value={item['value']:.4f}")
        normality_json_path = output_dir / f"{case_name}_normality.json"
        normality_json_path.write_text(
            json.dumps(normality_result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"✓ 常模判别结果已保存: {normality_json_path}")
    
    if labels_only:
        return prediction, normality_result

    # 可视化多个切片
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    n_slices = img_arr.shape[0]
    slice_indices = [n_slices // 4, n_slices // 2, 3 * n_slices // 4]
    
    for slice_idx in slice_indices:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 原始图像
        axes[0].imshow(img_arr[slice_idx], cmap='gray')
        axes[0].set_title(f'Original Image (Slice {slice_idx})')
        axes[0].axis('off')
        
        # 预测结果
        axes[1].imshow(prediction[slice_idx], cmap='tab10', vmin=0, vmax=7)
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        # 添加颜色图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=plt.cm.tab10(i/10), label=MMWHS_CLASS_NAMES[i]) 
                          for i in range(min(8, len(MMWHS_CLASS_NAMES)))]
        fig.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout()
        save_path = vis_dir / f"{case_name}_slice_{slice_idx:03d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 保存可视化: {save_path.name}")
    
    return prediction, normality_result


def predict_batch(
    model,
    data_dir,
    modality,
    device,
    output_dir,
    normal_model=None,
    labels_only=False,
    patch_size=(64, 128, 128),
    stride=(32, 64, 64),
):
    """批量预测（无需标签）"""
    model.eval()
    
    # 查找图像文件
    data_dir = Path(data_dir)
    test_pattern = f"{modality}_test_*_image.nii.gz"
    
    test_files = []
    for search_pattern in [f"*_{modality}_test/{test_pattern}", 
                          f"{modality}_test/{test_pattern}", 
                          test_pattern]:
        test_files = sorted(list(data_dir.glob(search_pattern)))
        if test_files:
            break
    
    if not test_files:
        raise FileNotFoundError(
            f"未找到测试图像！\n"
            f"搜索路径: {data_dir}\n"
            f"搜索模式: {test_pattern}"
        )
    
    print(f"\n找到 {len(test_files)} 个测试样本")
    print(f"数据目录: {data_dir}\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 预测每个样本
    for idx, image_file in enumerate(test_files):
        print(f"\n[{idx+1}/{len(test_files)}] {'='*50}")
        predict_single_image(
            model,
            str(image_file),
            device,
            output_dir,
            normal_model=normal_model,
            labels_only=labels_only,
            patch_size=patch_size,
            stride=stride,
        )
    
    print(f"\n{'='*60}")
    print(f"✓ 所有预测完成！结果保存在: {output_dir}")
    print(f"{'='*60}")


def predict_batch_by_glob(
    model,
    glob_root,
    input_glob,
    device,
    output_dir,
    normal_model=None,
    labels_only=False,
    patch_size=(64, 128, 128),
    stride=(32, 64, 64),
):
    """按自定义 glob 模式批量预测"""
    model.eval()

    root = Path(glob_root)
    if not root.exists():
        raise FileNotFoundError(f"glob_root 不存在: {root}")

    test_files = sorted(list(root.glob(input_glob)))
    if not test_files:
        raise FileNotFoundError(
            f"未找到匹配图像！\n"
            f"搜索根目录: {root}\n"
            f"搜索模式: {input_glob}"
        )

    print(f"\n找到 {len(test_files)} 个测试样本")
    print(f"搜索根目录: {root}")
    print(f"搜索模式: {input_glob}\n")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, image_file in enumerate(test_files):
        print(f"\n[{idx+1}/{len(test_files)}] {'='*50}")
        predict_single_image(
            model,
            str(image_file),
            device,
            output_dir,
            normal_model=normal_model,
            labels_only=labels_only,
            patch_size=patch_size,
            stride=stride,
        )

    print(f"\n{'='*60}")
    print(f"✓ 所有预测完成！结果保存在: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="MRI 模型推理（无需标签）")
    
    # 模型
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型checkpoint路径")
    parser.add_argument("--base_channels", type=int, default=32,
                        help="模型基础通道数")
    
    # 输入：单个文件或批量目录
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str,
                       help="单个图像路径 (*_image.nii.gz)")
    group.add_argument("--data_dir", type=str,
                       help="批量预测：数据根目录")
    group.add_argument("--input_glob", type=str,
                       help="批量预测：自定义glob模式（例: mr_train/mr_train_*_image.nii.gz）")
    
    parser.add_argument("--modality", type=str, default="mr",
                        help="模态 (mr 或 ct)")
    parser.add_argument("--output_dir", type=str, default="predictions",
                        help="结果保存目录")
    parser.add_argument("--glob_root", type=str, default=".",
                        help="自定义glob的搜索根目录（仅 --input_glob 生效）")
    parser.add_argument("--device", type=str, default="cuda",
                        help="计算设备")
    parser.add_argument("--patch_size", type=int, nargs=3, default=[64, 128, 128],
                        help="滑窗patch大小 D H W")
    parser.add_argument("--stride", type=int, nargs=3, default=[32, 64, 64],
                        help="滑窗步长 D H W")
    parser.add_argument("--labels_only", action="store_true",
                        help="仅保存分割标签，不生成可视化和常模结果")
    parser.add_argument("--normal_model", type=str, default="",
                        help="可选：常模模型json路径，提供后会进行第二模型判别")
    
    args = parser.parse_args()
    
    # 检查模型文件
    if not Path(args.checkpoint).exists():
        print(f"错误: 模型文件不存在: {args.checkpoint}")
        return
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"使用设备: {device}")
    print(f"{'='*60}")
    
    # 加载模型
    print("\n加载模型...")
    model = get_model(num_classes=8, base_channels=args.base_channels)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    if 'best_dice' in checkpoint:
        print(f"模型训练最佳Dice: {checkpoint['best_dice']:.4f}")
    if 'epoch' in checkpoint:
        print(f"训练轮数: {checkpoint['epoch']}")
    
    # 常模模型（可选）
    normal_model = None
    if args.normal_model and (not args.labels_only):
        normal_path = Path(args.normal_model)
        if not normal_path.exists():
            print(f"错误: 常模模型不存在: {normal_path}")
            return
        normal_model = load_normal_heart_model(normal_path)
        print(f"常模模型加载成功: {normal_path}")
    elif args.normal_model and args.labels_only:
        print("labels_only 已启用，跳过常模模型判别。")

    # 预测
    if args.image:
        # 单个文件预测
        if not Path(args.image).exists():
            print(f"错误: 图像文件不存在: {args.image}")
            return
        
        _prediction, _normality = predict_single_image(
            model,
            args.image,
            device,
            args.output_dir,
            normal_model=normal_model,
            labels_only=args.labels_only,
            patch_size=tuple(args.patch_size),
            stride=tuple(args.stride),
        )
    elif args.input_glob:
        # 自定义glob批量预测
        predict_batch_by_glob(
            model=model,
            glob_root=args.glob_root,
            input_glob=args.input_glob,
            device=device,
            output_dir=args.output_dir,
            normal_model=normal_model,
            labels_only=args.labels_only,
            patch_size=tuple(args.patch_size),
            stride=tuple(args.stride),
        )
    else:
        # 批量预测
        predict_batch(
            model,
            args.data_dir,
            args.modality,
            device,
            args.output_dir,
            normal_model=normal_model,
            labels_only=args.labels_only,
            patch_size=tuple(args.patch_size),
            stride=tuple(args.stride),
        )
    
    print(f"\n✓ 完成！结果保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
