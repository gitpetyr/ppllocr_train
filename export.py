from ultralytics import YOLO
import os

_MODEL_CANDIDATES = [
    "models/v2_beta1218/yolo11m_universal_final/weights/best.pt",
    "models/v2_beta1218/yolo11m_universal_final/weights/last.pt",
    "models/v1/weights/best.pt",
    "models/v1/weights/last.pt",
]
MODEL_PATH = next((p for p in _MODEL_CANDIDATES if os.path.exists(p)), _MODEL_CANDIDATES[0])
EXPORT_PATH = "models/onnx/ppllocr_betav2.onnx"

def export_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型: {MODEL_PATH}")
        print("可选路径示例：")
        for p in _MODEL_CANDIDATES:
            print(f"  - {p}")
        return

    print(f"🚀 导出灵活版模型 (无NMS, size=512): {MODEL_PATH}")
    
    model = YOLO(MODEL_PATH)
    
    # 关键：不加 nms=True
    exported = model.export(
        format="onnx", 
        dynamic=True, 
        simplify=True,
        imgsz=512,      # 依然建议锁定训练尺寸，防止特征不对齐
        opset=12        # 保持兼容性
    )
    
    if not exported:
        print("❌ 导出失败")
        return

    exported_path = str(exported)
    if not os.path.exists(exported_path):
        print(f"❌ 未找到导出产物: {exported_path}")
        return

    os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
    os.replace(exported_path, EXPORT_PATH)
    print(f"✅ 导出成功: {EXPORT_PATH}")

if __name__ == "__main__":
    export_model()
