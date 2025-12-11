import os
import random
import string
import shutil
import math
import glob
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageStat, ImageChops, ImageOps
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, freeze_support

# ==========================================
# 1. 全局配置 (Configuration)
# ==========================================

OUTPUT_DIR = "dataset_universal_final"

# 尺寸配置
CAPTCHA_W_RANGE = (90, 360)
CAPTCHA_H_RANGE = (35, 120)

# 字符集 (保持不变)
SPECIFIC_SYMBOLS = "/*%@#"
CHARACTERS = string.digits + string.ascii_letters + SPECIFIC_SYMBOLS
CHAR_MAP = {char: i for i, char in enumerate(CHARACTERS)}
CLASS_COUNT = len(CHARACTERS)

# 生成数量
NUM_TRAIN = 100000 
NUM_VAL = 2000

# [调整] 样本难度分布 (增加普通样本，减少变态样本)
PROB_NEGATIVE = 0.15   
PROB_NORMAL   = 0.30   # 提升到 50%
PROB_HARD     = 0.55   # 降至 40%

# 资源路径
FONTS_DIR = "fonts"
BG_DIR = "backgrounds"

# ==========================================
# 2. 核心工具库
# ==========================================

FONT_FILES = []
BG_FILES = []

def init_worker(fonts, bgs):
    global FONT_FILES, BG_FILES
    FONT_FILES = fonts
    BG_FILES = bgs
    random.seed() 
    np.random.seed()

def load_resources_paths():
    fonts = glob.glob(os.path.join(FONTS_DIR, "*.[ot]tf"))
    bgs = glob.glob(os.path.join(BG_DIR, "*.*"))
    bgs = [f for f in bgs if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    return fonts, bgs

def get_random_font(size):
    if not FONT_FILES: return ImageFont.load_default()
    try:
        return ImageFont.truetype(random.choice(FONT_FILES), max(10, int(size)))
    except:
        return ImageFont.load_default()

def random_color(start=0, end=255, alpha=None):
    r = random.randint(start, end)
    g = random.randint(start, end)
    b = random.randint(start, end)
    if alpha is not None:
        return (r, g, b, alpha)
    return (r, g, b)

def get_color_scheme(style="normal"):
    is_dark_mode = random.random() < 0.5
    if is_dark_mode:
        bg_range = (0, 80); fg_range = (160, 255) # 对比度稍微拉大一点
    else:
        bg_range = (180, 255); fg_range = (0, 90)
    bg_color = random_color(*bg_range)
    if style == "hollow":
        text_fill = bg_color
        text_stroke = random_color(*fg_range)
    else:
        text_fill = random_color(*fg_range)
        text_stroke = None
    return bg_color, text_fill, text_stroke

# ==========================================
# 3. 高级扭曲算法 (削弱版)
# ==========================================

def cv2_elastic_transform(img_pil, alpha=1000, sigma=30):
    img = np.array(img_pil)
    h, w = img.shape[:2]
    # [削弱] 扭曲力度减半，保留文字骨架
    dx = cv2.GaussianBlur(np.random.rand(h, w)*2-1, (0, 0), sigma) * (alpha * 0.6)
    dy = cv2.GaussianBlur(np.random.rand(h, w)*2-1, (0, 0), sigma) * (alpha * 0.6)
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)
    distorted = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return Image.fromarray(distorted)

def cv2_fisheye(img_pil, mode="barrel"):
    img = np.array(img_pil)
    h, w = img.shape[:2]
    K = np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]], dtype=np.float32)
    # [削弱] 畸变参数减小
    if mode == "barrel":
        D = np.array([random.uniform(0.05, 0.2), random.uniform(0.05, 0.2), 0, 0], dtype=np.float32)
    else:
        D = np.array([random.uniform(-0.15, -0.05), random.uniform(-0.15, -0.05), 0, 0], dtype=np.float32)
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w,h), 1, (w,h))
    distorted = cv2.undistort(img, K, D, None, new_K)
    return Image.fromarray(distorted)

def cv2_perspective(img_pil):
    img = np.array(img_pil)
    h, w = img.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    # [削弱] 透视拉伸幅度减小
    def rand_off(): return random.randint(int(min(w,h)*0.02), int(min(w,h)*0.10))
    dst_pts = np.float32([[rand_off(), rand_off()], [w - rand_off(), rand_off()], [rand_off(), h - rand_off()], [w - rand_off(), h - rand_off()]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    distorted = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    return Image.fromarray(distorted)

def apply_chromatic_aberration(image, offset=3):
    r, g, b = image.split()
    r = ImageChops.offset(r, random.randint(-offset, offset), random.randint(-offset, offset))
    b = ImageChops.offset(b, random.randint(-offset, offset), random.randint(-offset, offset))
    return Image.merge("RGB", (r, g, b))

# ==========================================
# 4. 靶向随机轨迹生成 (削弱版)
# ==========================================

def calculate_bezier_point(t, p0, p1, p2, p3):
    u = 1 - t
    tt = t * t
    uu = u * u
    u3 = uu * u
    t3 = tt * t
    x = u3 * p0[0] + 3 * uu * t * p1[0] + 3 * u * tt * p2[0] + t3 * p3[0]
    y = u3 * p0[1] + 3 * uu * t * p1[1] + 3 * u * tt * p2[1] + t3 * p3[1]
    return (x, y)

def draw_random_chaos_curve(draw, p1, p2, fill, width):
    """
    [削弱版] 混沌轨迹：更平滑，没那么乱
    """
    x1, y1 = p1
    x2, y2 = p2
    
    # 减少分段，线条更流畅
    segments = random.randint(1, 3) 
    
    points = []
    points.append((x1, y1))
    
    for i in range(1, segments):
        t = i / segments
        base_x = x1 + (x2 - x1) * t
        base_y = y1 + (y2 - y1) * t
        
        # 减少随机抖动幅度
        dist = math.hypot(x2-x1, y2-y1)
        offset_range = dist * 0.15 # 从 30% 降到 15%
        
        rx = base_x + random.uniform(-offset_range, offset_range)
        ry = base_y + random.uniform(-offset_range, offset_range)
        points.append((rx, ry))
        
    points.append((x2, y2))
    
    final_path_points = []
    
    for i in range(len(points) - 1):
        start_p = points[i]
        end_p = points[i+1]
        
        seg_dist = math.hypot(end_p[0]-start_p[0], end_p[1]-start_p[1])
        ctrl_range = seg_dist * random.uniform(0.2, 0.8) # 减少控制点发散
        
        cp1 = (start_p[0] + random.uniform(-ctrl_range, ctrl_range), 
               start_p[1] + random.uniform(-ctrl_range, ctrl_range))
        
        cp2 = (end_p[0] + random.uniform(-ctrl_range, ctrl_range), 
               end_p[1] + random.uniform(-ctrl_range, ctrl_range))
        
        steps = max(10, int(seg_dist / 2))
        for t in range(steps):
            pt = calculate_bezier_point(t/steps, start_p, cp1, cp2, end_p)
            final_path_points.append(pt)
            
    final_path_points.append(points[-1])
    
    if len(final_path_points) > 1:
        draw.line(final_path_points, fill=fill, width=width, joint='curve')


def draw_targeted_interference(draw, w, h, labels, style="curve"):
    if not labels: return

    # [削弱] 动态线宽变细 (3% - 5%)
    base_width = max(1, int(h * 0.03)) 
    
    # 减少线条数量
    num_lines = random.randint(1, 3)
    
    for _ in range(num_lines):
        strategy = random.choice(["stitch", "cut_corner", "edge_scratch"])
        
        # [削弱] 透明度增加 (更透)，减少遮挡
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255), random.randint(100, 200))
        width = random.randint(base_width, base_width + 2)
        
        if strategy == "stitch" and len(labels) > 1:
            idx = random.randint(0, len(labels) - 2)
            box1 = labels[idx]
            box2 = labels[idx+1]
            start_p = (random.randint(int(box1[1]), int(box1[3])), random.randint(int(box1[2]), int(box1[4])))
            end_p = (random.randint(int(box2[1]), int(box2[3])), random.randint(int(box2[2]), int(box2[4])))
            draw_random_chaos_curve(draw, start_p, end_p, color, width)
            
        elif strategy == "cut_corner":
            target_box = random.choice(labels)
            bx1, by1, bx2, by2 = target_box[1:]
            corner_x = random.choice([bx1, bx2])
            corner_y = random.choice([by1, by2])
            p1 = (corner_x + random.randint(-15, 15), corner_y + random.randint(-15, 15))
            p2 = (bx1 + (bx2-bx1)/2, by1 + (by2-by1)/2)
            draw_random_chaos_curve(draw, p1, p2, color, width)
            
        else:
            if random.random() < 0.5: 
                y_level = random.choice([random.randint(0, int(h*0.2)), random.randint(int(h*0.8), h)])
                p1 = (0, y_level); p2 = (w, y_level + random.randint(-10, 10))
            else: 
                x_level = random.choice([random.randint(0, int(w*0.3)), random.randint(int(w*0.7), w)])
                p1 = (x_level, 0); p2 = (x_level + random.randint(-10, 10), h)
            draw_random_chaos_curve(draw, p1, p2, color, width)

def draw_grid_mask(image, w, h, strength="hard"):
    grid_layer = Image.new('RGBA', (w, h), (0,0,0,0))
    d = ImageDraw.Draw(grid_layer)
    # [削弱] 网格更稀疏
    step = random.randint(25, 45) 
    # [削弱] 颜色更淡
    color = (random.randint(0,80), random.randint(0,80), random.randint(0,80), 160) 
    # [削弱] 线更细
    min_w = 1; max_w = max(2, int(h * 0.03))
    
    for x in range(0, w, step):
        d.line((x, 0, x, h), fill=color, width=random.randint(min_w, max_w))
    for y in range(0, h, step):
        d.line((0, y, w, y), fill=color, width=random.randint(min_w, max_w))
        
    if strength == "hard": grid_layer = cv2_fisheye(grid_layer, mode="barrel")
    image.paste(grid_layer, (0,0), grid_layer)
    return image

def draw_camouflage(draw, w, h, text_color, count=5):
    """[削弱] 伪装干扰"""
    stroke_w_base = max(2, int(h * 0.04)) # 笔画稍细
    for _ in range(count):
        x, y = random.randint(0, w), random.randint(0, h)
        width = random.randint(stroke_w_base, int(stroke_w_base * 1.5))
        if random.random() < 0.5:
            draw.ellipse((x, y, x+width, y+width), fill=text_color)
        else:
            length = random.randint(width * 2, width * 3)
            direction = random.choice([(1,0), (0,1), (1,1), (1,-1)])
            x2 = x + length * direction[0]
            y2 = y + length * direction[1]
            draw.line((x, y, x2, y2), fill=text_color, width=width)

    if isinstance(text_color, tuple) and len(text_color) >= 3:
        contrast_color = (255-text_color[0], 255-text_color[1], 255-text_color[2])
    else: contrast_color = (128, 128, 128)
        
    for _ in range(random.randint(1, 3)): # 减少异色切割数量
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)
        draw.line((x1, y1, x2, y2), fill=contrast_color, width=random.randint(1, 3))

def draw_heavy_clutter(image, font_base_size):
    w, h = image.size
    count = random.randint(10, 20) # 减少背景字数量
    for _ in range(count):
        char = random.choice(CHARACTERS)
        fs = int(font_base_size * random.uniform(0.5, 1.5))
        font = get_random_font(fs)
        # [削弱] 背景字更加透明 (20-60)
        color = (*random_color(0, 255), random.randint(20, 60))
        txt_layer = Image.new('RGBA', (fs*2, fs*2), (0,0,0,0))
        td = ImageDraw.Draw(txt_layer)
        td.text((fs, fs), char, font=font, fill=color, anchor="mm")
        txt_layer = txt_layer.rotate(random.randint(0, 360))
        cx, cy = random.randint(0, w), random.randint(0, h)
        image.paste(txt_layer, (cx, cy), txt_layer)
    return image.convert('RGB')

# ==========================================
# 5. 核心生成逻辑 (Pro Generator)
# ==========================================

def generate_captcha_pro(difficulty="normal"):
    if difficulty == "negative":
        h = random.randint(*CAPTCHA_H_RANGE)
        w = int(h * random.uniform(2.0, 4.0))
        bg_color = random_color(0, 255)
        image = Image.new('RGB', (w, h), bg_color)
        if random.random() < 0.5: image = draw_heavy_clutter(image, h//2)
        if random.random() < 0.5: draw_grid_mask(image, w, h)
        return image, []

    styles = ["classic", "shadow_hollow", "chromatic", "grid_fisheye", "geometric_chaos"]
    style = random.choice(styles)
    
    h = random.randint(*CAPTCHA_H_RANGE)
    w = int(h * random.uniform(2.5, 5.0))
    bg_color, text_fill, text_stroke = get_color_scheme("hollow" if "hollow" in style else "normal")
    
    image = Image.new('RGB', (w, h), bg_color)
    # 普通难度下减少 Clutter 出现概率
    if difficulty == "hard" or random.random() < 0.2:
        image = draw_heavy_clutter(image, int(h*0.6))

    margin = 50
    temp_w, temp_h = w + margin*2, h + margin*2
    length = random.randint(4, 7)
    text = ''.join(random.choices(CHARACTERS, k=length))
    base_font_size = int(h * random.uniform(0.6, 0.9))
    start_x = margin + random.randint(10, 50)
    
    char_boxes = [] 
    
    for char in text:
        font = get_random_font(base_font_size + random.randint(-5, 5))
        bbox = font.getbbox(char)
        cw, ch = bbox[2]-bbox[0], bbox[3]-bbox[1]
        size = int(math.hypot(cw, ch)) + 10
        char_img = Image.new('RGBA', (size, size), (0,0,0,0))
        cd = ImageDraw.Draw(char_img)
        
        if style == "shadow_hollow":
            off = random.randint(2, 4)
            cd.text((size//2 + off, size//2 + off), char, font=font, fill=(100,100,100,120), anchor="mm")
            # [削弱] 描边不那么粗了
            cd.text((size//2, size//2), char, font=font, fill=None, stroke_width=random.randint(1, 3), stroke_fill=text_stroke or (0,0,0), anchor="mm")
        else:
            cd.text((size//2, size//2), char, font=font, fill=text_fill, anchor="mm")

        rot_angle = random.uniform(-20, 20) if difficulty == "hard" else random.uniform(-5, 5)
        char_img = char_img.rotate(rot_angle, resample=Image.BICUBIC)
        r_box = char_img.getbbox()
        if r_box:
            overlap = 0.85 if difficulty == "hard" else 0.98 # 增加间距
            paste_y = (temp_h // 2) - (r_box[3]-r_box[1])//2 + random.randint(-5, 5)
            layer = Image.new('RGBA', (temp_w, temp_h), (0,0,0,0))
            layer.paste(char_img, (start_x, paste_y), char_img)
            char_boxes.append({'id': CHAR_MAP[char], 'layer': layer})
            start_x += int(cw * overlap)

    merged_text_layer = Image.new('RGBA', (temp_w, temp_h), (0,0,0,0))
    for cb in char_boxes: merged_text_layer = Image.alpha_composite(merged_text_layer, cb['layer'])
        
    map_x, map_y = None, None
    if difficulty == "hard" or style == "geometric_chaos":
        # [削弱] 扭曲参数减小
        alpha = random.randint(600, 800) 
        sigma = random.randint(50, 65) # Sigma 变大一点，波纹更平滑
        dx = cv2.GaussianBlur(np.random.rand(temp_h, temp_w)*2-1, (0,0), sigma) * alpha
        dy = cv2.GaussianBlur(np.random.rand(temp_h, temp_w)*2-1, (0,0), sigma) * alpha
        x_mesh, y_mesh = np.meshgrid(np.arange(temp_w), np.arange(temp_h))
        map_x, map_y = np.float32(x_mesh + dx), np.float32(y_mesh + dy)
        
    def remap_image(pil_img, mx, my):
        if mx is None: return pil_img
        return Image.fromarray(cv2.remap(np.array(pil_img), mx, my, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0)))

    final_text_layer = remap_image(merged_text_layer, map_x, map_y)
    # [削弱] 减少透视出现的概率
    if style == "geometric_chaos" and map_x is None and random.random() < 0.5: 
        final_text_layer = cv2_perspective(final_text_layer)

    final_labels = []
    paste_x = (w - temp_w) // 2
    paste_y = (h - temp_h) // 2
    
    for cb in char_boxes:
        dist_char_layer = remap_image(cb['layer'], map_x, map_y)
        if style == "geometric_chaos" and map_x is None and random.random() < 0.5: 
            dist_char_layer = cv2_perspective(dist_char_layer)
        bbox = dist_char_layer.getbbox()
        if bbox:
            real_x1 = max(0, bbox[0] + paste_x); real_y1 = max(0, bbox[1] + paste_y)
            real_x2 = min(w, bbox[2] + paste_x); real_y2 = min(h, bbox[3] + paste_y)
            if real_x2 > real_x1 + 2 and real_y2 > real_y1 + 2:
                final_labels.append((cb['id'], real_x1, real_y1, real_x2, real_y2))

    image.paste(final_text_layer, (paste_x, paste_y), final_text_layer)
    d_final = ImageDraw.Draw(image)
    
    if style == "classic" or difficulty == "hard":
        draw_targeted_interference(d_final, w, h, final_labels)

    if difficulty == "hard":
        draw_camouflage(d_final, w, h, text_fill, count=random.randint(2, 5))
        
    if style == "grid_fisheye": image = draw_grid_mask(image, w, h, strength="hard")
    if style == "chromatic": image = apply_chromatic_aberration(image, offset=random.randint(1, 3))
    if random.random() < 0.2: image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 0.8)))

    return image, final_labels

# ==========================================
# 6. 任务调度
# ==========================================

def process_single_task(args):
    idx, img_dir, lbl_dir = args
    r = random.random()
    if r < PROB_NEGATIVE: difficulty = "negative"
    elif r < PROB_NEGATIVE + PROB_NORMAL: difficulty = "normal"
    else: difficulty = "hard"
        
    try:
        img, lbls = generate_captcha_pro(difficulty)
        name = f"{difficulty}_{idx:07d}"
        img.save(os.path.join(img_dir, name + ".jpg"), quality=random.randint(85, 95))
        img_w, img_h = img.size
        dw, dh = 1. / img_w, 1. / img_h
        lines = []
        for cid, x1, y1, x2, y2 in lbls:
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w/2.0, y1 + h/2.0
            cx, cy = min(max(cx*dw, 0), 1), min(max(cy*dh, 0), 1)
            nw, nh = min(max(w*dw, 0), 1), min(max(h*dh, 0), 1)
            if nw > 0.01 and nh > 0.01:
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as f: f.write("\n".join(lines))
        return True
    except: return False

def run_generation():
    freeze_support()
    if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
    dirs = {s: {'img': os.path.join(OUTPUT_DIR, "images", s), 'lbl': os.path.join(OUTPUT_DIR, "labels", s)} for s in ['train', 'val']}
    for s in dirs: os.makedirs(dirs[s]['img'], exist_ok=True); os.makedirs(dirs[s]['lbl'], exist_ok=True)

    print("Loading resources...")
    fonts, bgs = load_resources_paths()
    if not fonts: print("❌ No fonts found!"); return
    
    tasks = []
    for i in range(NUM_TRAIN): tasks.append((i, dirs['train']['img'], dirs['train']['lbl']))
    for i in range(NUM_VAL): tasks.append((i, dirs['val']['img'], dirs['val']['lbl']))

    print(f"Generating {len(tasks)} Optimized-Difficulty images...")
    with Pool(processes=max(1, cpu_count() * 2), initializer=init_worker, initargs=(fonts, bgs)) as pool:
        list(tqdm(pool.imap_unordered(process_single_task, tasks), total=len(tasks)))

    yaml_content = f"path: {os.path.abspath(OUTPUT_DIR)}\ntrain: images/train\nval: images/val\nnc: {CLASS_COUNT}\nnames: {list(CHARACTERS)}\n"
    with open(os.path.join(OUTPUT_DIR, "data.yaml"), "w") as f: f.write(yaml_content)
    print(f"\n✅ Dataset Generation Complete: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_generation()