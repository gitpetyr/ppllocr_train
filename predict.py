import cv2
import string
import torch
from ultralytics import YOLO

# ================= 配置区域 =================
# 1. 模型路径 (请修改为您训练好的 best.pt 路径)
MODEL_PATH = "runs/detect/yolo11m_universal_final/weights/last.pt" 

# 2. 测试图片路径 (可以是单张图片，也可以是文件夹)
SOURCE_PATH = "" 

# 3. 字符集 (必须与 data_gen_unified.py 中的完全一致！)
CHARACTERS = string.digits + string.ascii_letters + string.punctuation
# ===========================================

def get_sorted_text(results):
    """
    OCR 核心逻辑：将检测到的框按从左到右排序，还原字符串
    """
    text_results = []
    
    for r in results:
        boxes = r.boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, cls]
        
        if len(boxes) == 0:
            return ""

        # 1. 提取 (x1, class_id)
        # 这里做简单的单行排序。如果是多行文本，需要先按 Y 轴聚类，再按 X 轴排序。
        # 对于验证码/单行文本，直接按 x1 (索引0) 排序即可。
        sorted_boxes = sorted(boxes, key=lambda x: x[0]) 

        decoded_chars = []
        for box in sorted_boxes:
            cls_id = int(box[5])
            conf = box[4]
            if cls_id < len(CHARACTERS):
                char = CHARACTERS[cls_id]
                decoded_chars.append(char)
        
        text_results.append("".join(decoded_chars))
        
    return text_results[0] if text_results else ""

def main():
    # 1. 加载模型
    print(f"Loading model from {MODEL_PATH}...")
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
    results = model.predict(source=SOURCE_PATH, save=True, conf=0.55, iou=0.5)

    # 3. 解析结果
    predicted_text = get_sorted_text(results)
    
    print("-" * 30)
    print(f"OCR 识别结果: {predicted_text}")
    print("-" * 30)
    print(f"结果图片已保存到: {results[0].save_dir}")

    # 4. 显示图片 (如果在桌面环境)
    # result_img = results[0].plot()
    # cv2.imshow("Result", result_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
