import cv2
import numpy as np
import onnxruntime as ort
import string
import time
import os
import random

# ================= 配置 =================
SPECIFIC_SYMBOLS = "/*%@#"
CHARACTERS = string.digits + string.ascii_letters + SPECIFIC_SYMBOLS
ONNX_MODEL_PATH = "models/onnx/ppllocr_v1.onnx" 
# =======================================

class PureONNXPredictor:
    def __init__(self, model_path, use_gpu=True):
        self.class_names = CHARACTERS
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            print(f"🚀 模型加载成功: {model_path} | 设备: {ort.get_device()}")
        except Exception as e:
            print(f"❌ 模型加载失败，尝试仅使用 CPU... ({e})")
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        # 强制指定模型输入尺寸 (必须与导出时一致，这里是 512)
        self.img_size = (512, 512) 

    def letterbox(self, im, new_shape=(512, 512), color=(114, 114, 114)):
        """
        核心修复：保持长宽比的缩放 (Letterbox)
        """
        shape = im.shape[:2]  # current shape [height, width]
        
        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # 计算 padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        # 居中填充 (divide padding by 2)
        dw /= 2  
        dh /= 2

        # 缩放
        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        # 填充边框
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        
        # 返回处理后的图，以及 缩放比例r, x方向偏移, y方向偏移
        return im, r, (left, top)

    def preprocess(self, img_src):
        """
        预处理：Letterbox -> BGR2RGB -> Normalize -> Transpose -> Expand dims
        """
        # 1. 使用 Letterbox 替代简单的 cv2.resize
        image, ratio, (dw, dh) = self.letterbox(img_src, new_shape=self.img_size)
        
        # 2. BGR 转 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 3. 归一化 (0-255 -> 0.0-1.0)
        image = image.astype(np.float32) / 255.0
        
        # 4. HWC -> CHW
        image = image.transpose(2, 0, 1)
        
        # 5. Add Batch
        image = np.expand_dims(image, axis=0)
        
        # 保存这些参数用于后处理时的坐标还原
        meta = {'ratio': ratio, 'dw': dw, 'dh': dh}
        return image, meta

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2
        y[:, 1] = x[:, 1] - x[:, 3] / 2
        y[:, 2] = x[:, 0] + x[:, 2] / 2
        y[:, 3] = x[:, 1] + x[:, 3] / 2
        return y

    def nms_numpy(self, boxes, scores, iou_threshold):
        if len(boxes) == 0: return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
        return keep

    def postprocess(self, output, meta, conf_thres, iou_thres):
        predictions = np.squeeze(output[0]).T 
        scores = np.max(predictions[:, 4:], axis=1)
        keep_mask = scores > conf_thres
        predictions = predictions[keep_mask]
        scores = scores[keep_mask]
        
        if len(predictions) == 0: return "", []
        
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = self.xywh2xyxy(predictions[:, :4])
        
        # NMS 去重
        indices = self.nms_numpy(boxes, scores, iou_thres)
        final_boxes = boxes[indices]
        final_scores = scores[indices]
        final_ids = class_ids[indices]
        
        # === 核心修复：坐标还原 (去除 Letterbox 的影响) ===
        # 1. 减去 padding
        final_boxes[:, 0] -= meta['dw'] # x1
        final_boxes[:, 2] -= meta['dw'] # x2
        final_boxes[:, 1] -= meta['dh'] # y1
        final_boxes[:, 3] -= meta['dh'] # y2
        
        # 2. 除以缩放比例
        final_boxes /= meta['ratio']
        
        # 3. 排序 (从左到右)
        sorted_indices = np.argsort(final_boxes[:, 0])
        
        result_text = []
        details = []
        
        for idx in sorted_indices:
            cid = final_ids[idx]
            if cid < len(self.class_names):
                char = self.class_names[cid]
                result_text.append(char)
                details.append({
                    "char": char,
                    "conf": float(final_scores[idx]),
                    "box": final_boxes[idx].tolist()
                })
        return "".join(result_text), details

    def predict(self, input_source, conf=0.25, iou=0.45):
        img = None
        if isinstance(input_source, str):
            if os.path.exists(input_source): img = cv2.imread(input_source)
        elif isinstance(input_source, bytes):
            nparr = np.frombuffer(input_source, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(input_source, np.ndarray):
            img = input_source
            
        if img is None: return "", []

        # 1. 预处理 (含 Letterbox)
        input_tensor, meta = self.preprocess(img)
        
        # 2. 推理
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        
        # 3. 后处理 (含坐标还原)
        text, details = self.postprocess(outputs, meta, conf, iou)
        
        return text, details

# ================= 测试 =================
if __name__ == "__main__":
    if not os.path.exists(ONNX_MODEL_PATH):
        print("请先导出模型")
        exit()

    predictor = PureONNXPredictor(ONNX_MODEL_PATH)
    
    # 找个测试图
    test_img_path = "屏幕截图 2025-12-08 221112.png" # 请确保文件存在
    
    print(f"\n🎯 测试图片: {test_img_path}")

    # === 测试 1: 传入路径 ===
    print("--- Mode 1: Path ---")
    text, _ = predictor.predict(test_img_path)
    print(f"Result: {text}")

    # === 测试 2: 传入 Bytes (模拟网络请求) ===
    print("--- Mode 2: Bytes ---")
    with open(test_img_path, "rb") as f:
        img_bytes = f.read() # 读取为二进制
    
    t0 = time.time()
    text, _ = predictor.predict(img_bytes) # 直接传 bytes
    print(f"Result: {text}")
    print(f"Time: {(time.time()-t0)*1000:.2f} ms")