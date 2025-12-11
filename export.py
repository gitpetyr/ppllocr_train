from ultralytics import YOLO
import os

MODEL_PATH = "models/v1/weights/last.pt" 
EXPORT_PATH = "models/onnx/ppllocr_v1.onnx"

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
        imgsz=512,      # 依然建议锁定训练尺寸，防止特征不对齐
        opset=12        # 保持兼容性
    )
    
    if success:
        if os.path.exists(success):
            os.rename(success, EXPORT_PATH)
            print(f"✅ 导出成功: {EXPORT_PATH}")

if __name__ == "__main__":
    export_model()