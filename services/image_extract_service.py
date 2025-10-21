import os
from io import BytesIO

import faiss
import matplotlib.pyplot as plt
import numpy as np
import requests
from PIL import Image, ImageEnhance
from keras.applications.vgg19 import VGG19, preprocess_input  # Sử dụng VGG19 thay vì ResNet50
from rembg import remove
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

from constants import name
from helpers.file_helper import save_file
from lib.supabase_client import SupabaseClient
from services.jewelry_image_vector_service import JewelryImageVectorService

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


class ImageExtractService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ImageExtractService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Initialize Supabase client
        if not hasattr(self, 'initialized'):  # Check if already initialized
            self.supabase_client = SupabaseClient()
            # Load the extraction model
            self.model = self.get_modified_VGG19_model()
            self.jewelry_image_vector_service = JewelryImageVectorService()
            self.initialized = True

    def get_modified_VGG19_model(self):
        # Load the VGG19 model with pretrained weights from ImageNet
        vgg19_model = VGG19(input_shape=(224, 224, 3), weights="imagenet")

        # Create a new model that outputs the feature vector from the last fully connected layer
        # Lấy đầu vào từ VGG19
        input_tensor = vgg19_model.input

        # Add a Dense layer to produce a feature vector of size 2048
        fc1_output = vgg19_model.get_layer("fc1").output

        # Thêm một lớp Dense để giảm kích thước đầu ra xuống 2048
        # reduced_output = Dense(512, activation='relu')(fc1_output)

        # Tạo mô hình mới
        modified_model = Model(inputs=input_tensor, outputs=fc1_output)
        modified_model.summary()
        return modified_model

    def image_preprocess(self, img):
        img = img.resize((224, 224))
        img = img.convert("RGB")
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        return x

    def extract_vector(self, img):
        """
        Extracts a feature vector from an image.

        Parameters:
        - model: The pre-trained model used to extract features.
        - img: A PIL Image object.

        Returns:
        - vector: The normalized feature vector extracted from the image.
        """
        print("Processing extract vector")
        img_tensor = self.image_preprocess(img)
        # Extract features
        vector = self.model.predict(img_tensor)[0]
        # print(f'Extracted vector: {str(vector.tolist())}')
        # Normalize the vector (L2 normalization)
        vector = vector / np.linalg.norm(vector)
        # print(f'Extracted vector L2: {str(vector.tolist())}')
        return vector

    def download_and_process_image(self, image_url: str) -> np.ndarray:
        """Download image from Supabase storage and create embedding."""
        try:
            # Fetch image from URL
            response = requests.get(image_url)
            response.raise_for_status()  # Raises an error for bad status codes
            # Convert bytes to image
            img = Image.open(BytesIO(response.content)).convert('RGB')

            # Generate augmented images
            augmented_images = self.augment_image(img)

            # # Display original and augmented images
            # self.display_images(augmented_images)
            # Preprocess the image and extract vector
            # Extract vectors for all augmented images
            vectors = []
            for augmented_img in augmented_images:
                vector = self.extract_vector(augmented_img)
                vectors.append(vector)
                print(f"Extracted vector: {vector.shape}")

            return np.array(vectors)  # Return an array of vectors
        except requests.RequestException as req_err:
            raise Exception(f"Error downloading image: {req_err}")
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")

    def display_images(self, images, titles=None):
        """Display a list of images with optional titles."""
        n = len(images)

        # Tính số hàng và số cột
        cols = 7  # Số cột mà bạn muốn hiển thị
        rows = (n + cols - 1) // cols  # Tính số hàng cần thiết

        plt.figure(figsize=(20, 5 * rows))  # Tính kích thước của figure dựa trên số hàng

        for i in range(n):
            plt.subplot(rows, cols, i + 1)  # Sử dụng lưới để đặt hình ảnh
            plt.imshow(images[i])
            plt.axis('off')  # Ẩn trục

            if titles is not None:
                plt.title(titles[i], pad=20)

            # Thêm kích thước của mỗi hình ảnh bên dưới hình ảnh
            width, height = images[i].size
            plt.text(0.5, -0.1, f"Size: {width}x{height}", ha='center', va='top', transform=plt.gca().transAxes)

        plt.tight_layout(pad=3)  # Điều chỉnh bố cục để tránh chồng chéo, tăng khoảng cách giữa các subplot
        plt.subplots_adjust(hspace=0.5, bottom=0.25)
        plt.show()

    def augment_image(self, img):
        """Generate augmented images from the original image."""
        augmented_images = []
        original_width, original_height = img.size
        # Original image
        augmented_images.append(img)

        # Rotate the image
        for angle in [45, 140, 220, 315]:
            augmented_images.append(img.rotate(angle))

        # Adjust brightness
        for factor in [0.7, 0.85]:  # Darker, original, brighter
            enhancer = ImageEnhance.Brightness(img)
            augmented_images.append(enhancer.enhance(factor))

        # Flip the image
        augmented_images.append(img.transpose(Image.FLIP_LEFT_RIGHT))
        augmented_images.append(img.transpose(Image.FLIP_TOP_BOTTOM))

        # # Add random cropping (optional)
        # width, height = img.size
        # for _ in range(3):  # Generate 3 random crops
        #     left = random.randint(0, width // 4)
        #     top = random.randint(0, height // 4)
        #     right = random.randint(3 * width // 4, width)
        #     bottom = random.randint(3 * height // 4, height)
        #     cropped_img = img.crop((left, top, right, bottom))
        #     augmented_images.append(cropped_img.resize((224, 224)))  # Resize to original size

        # Add zoom in and zoom out
        zoom_factors = [0.5, 1.2, 1.5, 2.0]  # Zoom out, smaller, and zoom in
        for zoom in zoom_factors:
            # Calculate new size
            new_width = int(original_width * zoom)
            new_height = int(original_height * zoom)

            # Resize the image
            zoomed_img = img.resize((new_width, new_height), Image.LANCZOS)

            # Center crop back to original size
            left = (zoomed_img.width - original_width) // 2
            top = (zoomed_img.height - original_height) // 2
            right = left + original_width
            bottom = top + original_height
            cropped_zoomed_img = zoomed_img.crop((left, top, right, bottom))

            augmented_images.append(cropped_zoomed_img)
        return augmented_images

    def image_url_to_vector(self, image_url: str) -> np.ndarray:
        """Convert image URL to vector using model.Return an array of vectors after rotate, brightness, flip,
        zoom in and zoom out."""
        image_vectors = self.download_and_process_image(image_url)
        return image_vectors

    def save_vector_db(self, data: list):
        """Save vector data to Supabase."""
        response = self.jewelry_image_vector_service.save(data)
        return response

    def process_image(self, images: list):
        """Convert image URL to vector using model."""
        vectors = []
        image_ids = []
        data_db = []
        for image_data in images:
            print(f'image: {image_data["url"]}')
            image_url = image_data.get('url')
            image_id = image_data.get('id')
            jewelry_id = image_data.get('jewelryId')
            if self.jewelry_image_vector_service.is_extracted(image_id) is True:
                continue

            vectors = self.download_and_process_image(image_url)
            data = [
                {
                    "image_id": image_id,
                    "parent_id": image_id if index != 0 else None,  # Corrected syntax for conditional assignment
                    "vector": image_vector.tolist(),
                    "jewelry_id": jewelry_id,
                    "has_vector": True,
                    "active": True
                } for index, image_vector in enumerate(vectors)  # Ensure that index comes before image_vector
            ]
            self.save_vector_db(data)

            print(f"Saved {len(data)} vectors to Supabase.")

    def save_model(self, vectors, paths) -> None:
        # Create and save a FAISS index
        try:
            # Save vectors to files
            save_file(vectors, name.FILE_VECTORS)

            # Save path to files
            save_file(paths, name.FILE_PATHS)

            # Save the FAISS index to a file
            vector_dim = vectors.shape[1]  # Vector dimensionality
            index = faiss.IndexFlatL2(vector_dim)  # Create an L2 distance FAISS index
            index.add(vectors)  # Add vectors to the index
            faiss.write_index(index, name.FILE_FAISS_INDEX)
            print("Successfully saved FAISS index to file")

        except Exception as e:
            print(f"Error handling FAISS index: {str(e)}")

    def remove_bg_image(self, file_bytes):
        try:
            if isinstance(file_bytes, bytes):
                img = Image.open(BytesIO(file_bytes)).convert('RGB')
                # return img
            else:
                raise TypeError("Expected bytes, got {}".format(type(file_bytes)))
            image_no_bg = remove(img)
            # image_no_bg.show()
            return image_no_bg

        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")

    def read_image_url(self, url: str):
        img = Image.open(url).convert('RGB')

        return img

    def test_remove_image_local(self, url: str):

        img = Image.open(url).convert('RGB')
        # Chuyển đổi ảnh từ PIL thành bytes
        img_bytes = BytesIO()
        img.save(img_bytes, format='PNG')  # Hoặc 'JPEG' tùy theo định dạng cần
        img_bytes = img_bytes.getvalue()  # Lấy bytes từ buffer

        image_no_bg = self.remove_bg_image(img_bytes)


def main():
    image_extract_service = ImageExtractService()

    url = (
        "https://cdn.pnj.io/images/detailed/225/sp-gnddddw013763-nhan-kim-cuong-vang-trang-14k-pnj-my-first-diamond"
        ".png")

    response = requests.get(url)
    response.raise_for_status()  # Raises an error for bad status codes
    # img = Image.open(BytesIO(response.content)).convert('RGB')
    # Convert bytes to image
    # Kiểm tra xóa background
    # image_extract_service.test_image_local(url)

    # kiểm tra trích đặc trưng
    # image = image_extract_service.download_and_process_image(url)
    # print(image.shape)
    img = image_extract_service.remove_bg_image(response.content)

    images = image_extract_service.augment_image(img)

    image_extract_service.display_images(images)
    # vgg_vectors = image_extract_service.download_and_process_image(url)
    # resnet_vectors = image_extract_service.download_and_process_image(url)
    #
    # # In kích thước vectors
    # print("VGG19 vector shape:", vgg_vectors.shape)  # Should be (13, 512)
    # print("ResNet50 vector shape:", resnet_vectors.shape)  # Should be (13, 512)
    #
    # # Tính độ tương đồng
    # for i in range(len(vgg_vectors)):
    #     similarity = np.dot(vgg_vectors[i], resnet_vectors[i])
    #     print(f"Độ tương đồng giữa VGG19 và ResNet50 vector thứ {i}: {similarity:.4f}")


if __name__ == "__main__":
    main()
