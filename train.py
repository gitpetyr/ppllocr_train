# train.py
import os
import torch
from ultralytics import YOLO

# --- 强制 MIOpen 使用立即模式，跳过耗时的搜索 ---
os.environ['MIOPEN_FIND_MODE'] = '1' 
os.environ['MIOPEN_USER_DB_PATH'] = '' # 禁用用户数据库以防损坏

def main():
    # -------------------------------------------------------------------------
    # 1. 硬件与环境配置
    # -------------------------------------------------------------------------
    # 检查是否有 GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 正在使用计算设备: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")
    # device="cpu"
    # 数据集配置文件路径 (对应您的 data_gen_unified.py 生成的目录)
    DATA_CONFIG = os.path.join('dataset_universal_final', 'data.yaml')

    # 安全检查
    if not os.path.exists(DATA_CONFIG):
        print(f"❌ 错误：未找到配置文件 '{DATA_CONFIG}'")
        print("   请先运行 'data_gen_unified.py' 生成数据集！")
        return

    # -------------------------------------------------------------------------
    # 2. 模型初始化
    # -------------------------------------------------------------------------
    # 使用 YOLOv11 Medium 版本
    # m 版比 s 版多一倍参数，对 94 类字符和复杂背景的特征提取能力更强
    print("📦 加载模型: yolo11m.pt ...")
    model = YOLO('runs/detect/yolo11m_universal_final/weights/last.pt') 

    # -------------------------------------------------------------------------
    # 3. 开始训练 (核心配置)
    # -------------------------------------------------------------------------
    print("🔥 开始训练通用 OCR 模型...")
    print("   目标: 验证码 + 通用文档识别 (对标 ddddocr)")
    
    results = model.train(
        # --- 基础配置 ---
        data=DATA_CONFIG,
        epochs=150,          # 8万张图，100轮足够收敛 (甚至50轮效果就很好了)
        imgsz=512,           # 统一画布尺寸 512x512
        batch=16,            # [显存警告] m模型+512图比较吃显存。如果报错 OOM，请改为 8 或 4
        device=device,
        name='yolo11m_universal_final', # 训练结果保存目录名
        workers=16,           # 数据加载线程数 (Windows下如果卡住不动，请改为 0)
        amp=True,
        # --- 损失函数权重 (根据您的需求特别调整) ---
        box=7.5,             # 保持默认高权重，确保粘连字符能被切开
        cls=2.5,             # [重点] 从 0.5 提至 2.4。强迫模型区分形似字 (0/O, 1/l/I)
        dfl=1.5,             # 辅助定位损失
        
        # --- 数据增强 (加载时在线增强) ---
        # 几何变换：模拟真实拍摄环境
        mosaic=1.0,          # [核心] 必须开启！让模型适应大图中的小文字
        
        degrees=6.0,         # 轻微旋转 (模拟相机歪斜)
        translate=0.1,       # 平移 (模拟构图偏差)
        scale=0.6,           # 缩放 (模拟远近变化，0.5 ~ 1.5倍)
        shear=0.0,           # 剪切 (生成器已做过强扭曲，这里关闭以免过度)
        perspective=0.0005,  # 透视 (模拟文档侧拍扫描)
        
        # 像素变换：模拟光照环境
        hsv_h=0.7,         # 色调微调
        hsv_s=0.72,           # 饱和度大幅波动 (模拟黑白/彩印/褪色)
        hsv_v=0.4,           # 亮度大幅波动 (模拟强光/暗光)
        
        # 混合增强
        mixup=0.1,           # 轻微混合 (模拟透底/水印干扰)
        
        # [死刑区] 绝对禁止的增强 (OCR 大忌)
        flipud=0.0,          # 禁止上下翻转 (6/9 不分)
        fliplr=0.0,          # 禁止左右翻转 (b/d 不分)
        copy_paste=0.0,      # 禁止复制粘贴 (破坏字间距语义)
        
        # --- 优化策略 ---
        optimizer='auto',    # v11 会自动选择 SGD 或 AdamW
        cos_lr=True,         # 使用余弦退火学习率，后期收敛更稳
        patience=20,         # 如果 20 轮没提升则提前停止
        resume=True
    )
    
    print(f"\n✅ 训练全部完成！")
    print(f"   最佳模型权重保存在: {results.save_dir}/weights/best.pt")
    print(f"   请查看: {results.save_dir}/val_batch_pred.jpg 分析实际效果")

if __name__ == '__main__':
    # Windows 下使用多进程必须放在 main 块中
    main()