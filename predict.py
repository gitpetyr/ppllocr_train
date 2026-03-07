import string
import os
from ultralytics import YOLO

# ================= 配置区域 =================
# 1. 模型路径 (请修改为您训练好的 best.pt 路径)
# 默认指向仓库内自带的 v2_beta1218 权重；如果你用 train.py 训练，则通常在 runs/ 下。
MODEL_PATH = "models/v2_beta1218/yolo11m_universal_final/weights/best.pt"

# 2. 测试图片路径 (可以是单张图片，也可以是文件夹)
SOURCE_PATH = "download.png" 

# 3. 字符集 (必须与 datagen.py 中的完全一致！)
SPECIFIC_SYMBOLS = "/*%@#+-()"
CHARACTERS = string.digits + string.ascii_letters + SPECIFIC_SYMBOLS
# ===========================================

def _get_char(cls_id: int, names):
    if isinstance(names, dict):
        name = names.get(cls_id)
        if isinstance(name, str) and name:
            return name

    if 0 <= cls_id < len(CHARACTERS):
        return CHARACTERS[cls_id]

    return ""

def get_sorted_text(result):
    """
    OCR 核心逻辑：将检测到的框按从左到右排序，还原字符串
    """
    boxes = result.boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, cls]

    if len(boxes) == 0:
        return ""

    # 1. 提取 (x1, class_id)
    # 这里做简单的单行排序。如果是多行文本，需要先按 Y 轴聚类，再按 X 轴排序。
    # 对于验证码/单行文本，直接按 x1 (索引0) 排序即可。
    sorted_boxes = sorted(boxes, key=lambda x: x[0])

    decoded_chars = []
    for box in sorted_boxes:
        cls_id = int(box[5])
        decoded_chars.append(_get_char(cls_id, getattr(result, "names", None)))

    return "".join(decoded_chars)

def main():
    # 1. 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        print("   你可以改为：")
        print("   - models/v1/weights/best.pt")
        print("   - runs/detect/yolo11m_universal_final/weights/best.pt")
        return
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("请检查路径是否正确，或者是否还在训练中。")
        return

    # 2. 预测
    # conf=0.25: 置信度阈值
    # iou=0.45: NMS 阈值，防止重叠框
    print(f"Predicting {SOURCE_PATH}...")
    results = model.predict(source=SOURCE_PATH, save=True, conf=0.5, iou=0.5)

    if not results:
        print("❌ 未返回任何预测结果")
        return

    # 3. 解析结果
    print("-" * 30)
    for result in results:
        predicted_text = get_sorted_text(result)
        source_path = getattr(result, "path", SOURCE_PATH)
        print(f"{source_path}: {predicted_text}")
    print("-" * 30)
    print(f"结果图片已保存到: {results[0].save_dir}")

    # 4. 显示图片 (如果在桌面环境)
    # result_img = results[0].plot()
    # cv2.imshow("Result", result_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
