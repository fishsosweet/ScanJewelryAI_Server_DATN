from ultralytics import YOLO
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import tempfile
import os

# Import hàm upload từ file khác
from services.cloudinary_service import upload_to_cloudinary  # đổi tên theo file thật

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
# 🔹 Ánh xạ tag YOLO sang tiếng Việt có dấu
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
# 🔹 Danh sách tag mô tả cho CLIP
# ====================================
TAG_CANDIDATES = {
    "vàng": "gold",
    "bạc": "silver",
    "kim cương": "diamond",
    "ngọc trai": "pearl",
    "thanh lịch": "elegant",
    "tối giản": "minimalist"
}


# ====================================
# 🔹 Hàm sinh tag bằng CLIP (trả về tiếng Việt)
# ====================================
def generate_clip_tags(image):
    """Sinh tag mô tả bằng CLIP (hiển thị tiếng Việt)"""
    try:
        vi_tags = list(TAG_CANDIDATES.keys())
        en_tags = list(TAG_CANDIDATES.values())

        inputs = clip_processor(
            text=en_tags,
            images=image,
            return_tensors="pt",
            padding=True
        ).to(device)

        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()[0]

        top_idx = probs.argsort()[-3:][::-1]
        top_tags_vi = [vi_tags[i] for i in top_idx]

        print(f"🧠 CLIP tags (vi): {top_tags_vi}")
        return top_tags_vi

    except Exception as e:
        print(f"❌ Lỗi sinh tag CLIP: {e}")
        return []


# ====================================
# 🔹 API /auto-tag
# ====================================
@app.route("/auto-tag", methods=["POST"])
def auto_tag():
    if not model:
        return jsonify({"error": "Model YOLO chưa được tải!"}), 500
    if not clip_model:
        return jsonify({"error": "Model CLIP chưa được tải!"}), 500

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "Không có file nào được gửi!"}), 400

    results_all = []

    for file in files:
        if file.filename == "":
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            file.save(tmp.name)
            image_path = tmp.name

        print(f"\n🔍 Xử lý ảnh: {file.filename}")

        try:
            # --- YOLO nhận dạng ---
            results = model(image_path, conf=0.5)
            names = model.names
            detected_tags = set()

            img = Image.open(image_path).convert("RGB")

            for r in results:
                if not r.boxes:
                    continue
                for box in r.boxes:
                    cls_id = int(box.cls)
                    tag_name = names.get(cls_id, "unknown")
                    tag_vi = YOLO_TAGS_VI.get(tag_name, tag_name)
                    detected_tags.add(tag_vi)
                    print(f"✅ Phát hiện: {tag_name} → {tag_vi}")

            if not detected_tags:
                detected_tags.add("không phát hiện sản phẩm")

            # --- CLIP sinh mô tả ---
            clip_tags = generate_clip_tags(img)
            detected_tags.update(clip_tags)

            # --- Upload lên Cloudinary ---
            cloud_url = upload_to_cloudinary(image_path, list(detected_tags))

            # --- Kết quả ---
            results_all.append({
                "filename": file.filename,
                "tags": list(detected_tags),
                "cloud_url": cloud_url
            })

        except Exception as e:
            print(f"❌ Lỗi khi xử lý {file.filename}: {e}")
            results_all.append({
                "filename": file.filename,
                "tags": ["Lỗi xử lý ảnh"],
                "cloud_url": None
            })

        finally:
            if os.path.exists(image_path):
                os.remove(image_path)

    print("\n🎯 Kết quả cuối cùng:", results_all)
    return jsonify({"results": results_all})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
