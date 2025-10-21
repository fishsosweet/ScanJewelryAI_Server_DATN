from ultralytics import YOLO
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageEnhance
from werkzeug.utils import secure_filename
import torch
import tempfile
import os
import json
import uuid

# Import hàm upload từ file khác
from services.cloudinary_service import upload_to_cloudinary

app = Flask(__name__)
CORS(app)

# ====================================
# 🔹 Nạp model YOLO
# ====================================
try:
    model = YOLO("runs/detect/train2/weights/best.pt")
    print("✅ Model YOLO đã được nạp thành công!")
except Exception as e:
    print(f"❌ Lỗi khi nạp model YOLO: {e}")
    model = None

# ====================================
# 🔹 Nạp model CLIP
# ====================================
try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model.to(device)
    print("✅ Model CLIP đã được nạp thành công!")
except Exception as e:
    print(f"❌ Lỗi khi nạp model CLIP: {e}")
    clip_model = None

# ====================================
# 🔹 Ánh xạ tag YOLO sang tiếng Việt
# ====================================
YOLO_TAGS_VI = {
    "Nhan": "nhẫn",
    "DayChuyen": "dây chuyền",
    "VongTay": "vòng tay",
    "BongTai": "bông tai",
    "MatDayChuyen": "mặt dây chuyền",
    "unknown": "không xác định"
}

# ====================================
# 🔹 Nhóm tag CLIP (chia theo loại)
# ====================================
TAG_GROUPS = {
    "chất liệu": {
        "vàng": "a gold jewelry piece",
        "bạc": "a silver jewelry piece",
    },
    "phong cách": {
        "tối giản": "a minimalist jewelry design",
        "thanh lịch": "an elegant jewelry style",
        "sang trọng": "a luxurious jewelry piece",
        "cổ điển": "a classic jewelry design",
        "hiện đại": "a modern jewelry style",
        "đáng yêu": "a cute jewelry item",
        "nữ tính": "a feminine jewelry piece",
        "nam tính": "a masculine jewelry style",
        "cá tính": "a bold jewelry design",
        "nghệ thuật": "an artistic jewelry design",
        "đính đá": "a gemstone jewelry"
    },
    "dịp sử dụng": {
        "đám cưới": "a wedding jewelry",
        "hẹn hò": "a date jewelry",
        "dự tiệc": "a party jewelry",
        "quà tặng": "a gift jewelry",
        "hằng ngày": "a daily wear jewelry",
        "sang trọng": "a formal jewelry",
        "công sở": "an office jewelry",
        "du lịch": "a travel jewelry"
    }
}

# ====================================
# 🔹 Hàm tăng màu cho CLIP
# ====================================
def enhance_image_for_clip(image: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(1.3)

# ====================================
# 🔹 Hàm sinh tag bằng CLIP
# ====================================
def generate_clip_tags(image: Image.Image):
    all_best_tags = []
    image = enhance_image_for_clip(image)

    for group_name, tag_dict in TAG_GROUPS.items():
        try:
            vi_texts = list(tag_dict.keys())
            en_prompts = list(tag_dict.values())

            inputs = clip_processor(
                text=en_prompts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(device)

            with torch.no_grad():
                outputs = clip_model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]

            best_idx = probs.argmax()
            best_vi = vi_texts[best_idx]
            all_best_tags.append(best_vi)

            print(f"🎯 CLIP ({group_name}): {best_vi} ({probs[best_idx]:.2f})")

        except Exception as e:
            print(f"❌ Lỗi sinh tag CLIP cho nhóm {group_name}: {e}")

    return all_best_tags


# ====================================
# 🔹 API 1: /auto-tag (chỉ phân tích, chưa upload)
# ====================================
@app.route("/auto-tag", methods=["POST"])
def auto_tag():
    if not model or not clip_model:
        return jsonify({"error": "Model chưa được nạp!"}), 500

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Không có file nào được gửi!"}), 400

    results_all = []

    for file in files:
        if file.filename == "":
            continue

        tmp_name = f"{uuid.uuid4().hex}.jpg"
        tmp_path = os.path.join(tempfile.gettempdir(), tmp_name)
        file.save(tmp_path)

        print(f"\n🔍 Xử lý ảnh: {file.filename}")

        try:
            results = model(tmp_path, conf=0.5)
            names = model.names
            detected_tags = set()
            img = Image.open(tmp_path).convert("RGB")

            for r in results:
                if not r.boxes:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls)
                    tag_name = names.get(cls_id, "unknown")
                    tag_vi = YOLO_TAGS_VI.get(tag_name, tag_name)
                    detected_tags.add(tag_vi)
                    print(f"✅ YOLO phát hiện: {tag_vi}")

            if not detected_tags:
                detected_tags.add("không phát hiện sản phẩm")

            # CLIP sinh mô tả
            clip_tags = generate_clip_tags(img)
            detected_tags.update(clip_tags)

            # Không upload — chỉ trả về path tạm
            results_all.append({
                "filename": file.filename,
                "tags": list(detected_tags),
                "temp_path": tmp_path
            })

        except Exception as e:
            print(f"❌ Lỗi xử lý {file.filename}: {e}")
            results_all.append({
                "filename": file.filename,
                "tags": ["Lỗi xử lý ảnh"],
                "temp_path": None
            })

    return jsonify({"results": results_all})


# ====================================
# 🔹 API 2: /upload-cloud (khi nhấn Lưu)
# ====================================

if __name__ == "__main__":
    app.run(debug=True, port=5001)
