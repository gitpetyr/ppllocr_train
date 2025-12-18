from ultralytics import YOLO
import os

MODEL_PATH = "models/beta_v2/yolo11m_universal_final/weights/best.pt" 
EXPORT_PATH = "models/onnx/ppllocr_betav2.onnx"

def export_model():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型: {MODEL_PATH}")
        return

    print(f"🚀 导出灵活版模型 (无NMS, size=512): {MODEL_PATH}")
    
    model = YOLO(MODEL_PATH)
    
    # 关键：不加 nms=True
    success = model.export(
        format="onnx", 
        dynamic=True, 
        simplify=True,
        imgsz=512,      # 依然建议锁定训练尺寸，防止特征不对齐
        opset=12        # 保持兼容性
    )
    
    if success:
        if os.path.exists(success):
            os.rename(success, EXPORT_PATH)
            print(f"✅ 导出成功: {EXPORT_PATH}")

if __name__ == "__main__":
    export_model()