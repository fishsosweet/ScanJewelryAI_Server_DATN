import logging
import cloudinary
import cloudinary.uploader
import os
from dotenv import load_dotenv
load_dotenv()

CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME')
CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY')
CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET')
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)
def upload_to_cloudinary(image_path, tags=None):
    folder = 'TrangSuc'
    try:
        response = cloudinary.uploader.upload(
            image_path,
            folder=folder,
            tags=tags if tags else []
        )
        return response.get("secure_url")
    except Exception as e:
        logging.error(f"Error uploading image to Cloudinary: {e}")
        return None


def search_images_by_keywords(keywords):
    from collections import defaultdict
    import cloudinary

    image_count = defaultdict(int)

    # Danh sách từ khóa chính
    main_keywords = ["nhẫn", "vòng", "bông tai", "dây chuyền", "lắc tay", "vòng cổ", "mặt dây chuyền", "lắc chân", "miếng","lắc"]

    # Tìm từ khóa chính đầu tiên xuất hiện
    main_kw = None
    for kw in keywords:
        for main in main_keywords:
            if main == kw:  # chỉ khớp chính xác
                main_kw = main
                break
        if main_kw:
            break

    # Nếu có từ khóa chính thì chỉ tìm đúng tag đó
    if main_kw:
        try:
            print(f"Tìm ảnh theo tag chính xác: {main_kw}")
            result = cloudinary.Search().expression(f"tags={main_kw}").execute()
            for item in result.get("resources", []):
                url = item["secure_url"]
                image_count[url] += 1
        except Exception as e:
            print(f"Lỗi khi tìm {main_kw}: {e}")
    # Sắp xếp ảnh theo số tag trùng giảm dần
    sorted_images = sorted(image_count.items(), key=lambda x: x[1], reverse=True)

    # Chỉ lấy tối đa 5 ảnh
    return [url for url, count in sorted_images][:5]

if __name__ == "__main__":
    url = upload_to_cloudinary("C:/Users/Fishh/Pictures/NhanCuoi/Lắc/Lactayco3lavang.jpg")
    print("URL ảnh trên Cloudinary:", url)
     
