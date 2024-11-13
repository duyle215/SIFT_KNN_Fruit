# app.py

import streamlit as st
import cv2
import numpy as np
import os
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog, graycomatrix, graycoprops
from sympy.printing.tensorflow import tensorflow

# Đường dẫn tới thư mục dữ liệu
DATA_DIR = 'D:\HUFLIT\ComputerVision\Project\SIFTProject\\fruit\images/'

# Danh sách các nhãn và mã hóa
labels_dict = {'apple fruit': 0, 'banana fruit': 1, 'cherry fruit': 2, 'chickoo fruit': 3,
               'grapes fruit': 4, 'kiwi fruit': 5, 'mango fruit': 6, 'orange fruit': 7, 'strawberry fruit': 8}  # Thêm các loại trái cây khác nếu có

# 1. Tiền xử lý Dữ liệu (Chuẩn hóa và Tăng cường)

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0  # Chuẩn hóa pixel về khoảng [0, 1]
    return img

import tensorflow as tf

# Tăng cường dữ liệu
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.2),         # Tương tự rotation_range=20
    tf.keras.layers.RandomBrightness(factor=0.2), # Tương tự brightness_range=[0.8, 1.2]
    tf.keras.layers.RandomZoom(0.2),              # Tương tự zoom_range=0.2
    tf.keras.layers.RandomFlip("horizontal_and_vertical") # Tương tự horizontal_flip và vertical_flip
])

def augment_image(img):
    aug_images = [data_augmentation(img, training=True) for _ in range(4)]  # Tạo 4 biến thể
    return aug_images

# 2. Trích xuất Đặc trưng

# 2.1. Trích xuất đặc trưng SIFT
def extract_sift_features(img):
    gray = cv2.cvtColor((img * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    return descriptors

# 2.2. Trích xuất đặc trưng HOG
def extract_hog_features(img):
    gray = cv2.cvtColor((img * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True)
    return features

# 2.3. Trích xuất Color Histogram
def extract_color_histogram(img, bins=(8, 8, 8)):
    # Ensure img is a NumPy array and in uint8 format for OpenCV
    if isinstance(img, tf.Tensor):
        img = img.numpy()  # Convert EagerTensor to NumPy array if needed
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)  # Convert to uint8 if img is in [0, 1] range

    # Calculate histogram
    hist = cv2.calcHist([img], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


# 2.4. Trích xuất đặc trưng Texture (GLCM)
def extract_texture_features(img):
    gray = cv2.cvtColor((img * 255).astype('uint8'), cv2.COLOR_RGB2GRAY)
    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256,
                        symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    ASM = graycoprops(glcm, 'ASM')[0, 0]
    return [contrast, dissimilarity, homogeneity, energy, correlation, ASM]

# 2.5. Kết hợp tất cả các đặc trưng
def extract_features(img):
    sift_desc = extract_sift_features(img)
    hog_feat = extract_hog_features(img)
    color_hist = extract_color_histogram(img)
    texture_feat = extract_texture_features(img)

    # Xử lý đặc trưng SIFT (sử dụng trung bình)
    sift_feat = np.mean(sift_desc, axis=0) if sift_desc is not None else np.zeros(128)

    # Kết hợp tất cả đặc trưng thành một vector
    features = np.hstack([sift_feat, hog_feat, color_hist, texture_feat])
    return features

# 3. Tạo tập dữ liệu
def create_dataset():
    X = []
    y = []
    for label in os.listdir(DATA_DIR):
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_dir):
            continue
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            img = load_and_preprocess_image(img_path)
            if img is None:
                continue
            # Tăng cường dữ liệu
            aug_images = augment_image(img)
            for aug_img in aug_images:
                features = extract_features(aug_img)
                X.append(features)
                y.append(labels_dict[label])
    return np.array(X), np.array(y)

# 4. Huấn luyện mô hình KNN
def train_model():
    X_train, y_train = create_dataset()

    #if X_train.ndim == 1:
    #    X_train = X_train.reshape(-1, 1)
    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Chọn tham số K và khoảng cách
    param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'metric': ['euclidean', 'manhattan', 'cosine']
    }

    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
    grid.fit(X_train_scaled, y_train)

    print("Best parameters:", grid.best_params_)
    print("Best cross-validation accuracy:", grid.best_score_)

    # Lưu mô hình và scaler
    joblib.dump(grid.best_estimator_, 'knn_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')

# Huấn luyện mô hình nếu chưa tồn tại
if not os.path.exists('knn_model.joblib') or not os.path.exists('scaler.joblib'):
    st.write("Đang huấn luyện mô hình, vui lòng đợi...")
    train_model()
    st.write("Huấn luyện mô hình hoàn tất.")

# 5. Xây dựng Ứng dụng Dự đoán với Streamlit

# Tải mô hình và scaler
model = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

# Mapping nhãn
labels_reverse_dict = {v: k for k, v in labels_dict.items()}

# Hàm tiền xử lý và trích xuất đặc trưng cho ảnh đầu vào
def preprocess_and_extract(img):
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    features = extract_features(img)
    return features

# Giao diện người dùng
st.title("Nhận diện trái cây bằng thuật toán SIFT")

uploaded_file = st.file_uploader("Chọn một ảnh trái cây", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, channels="BGR", caption="Ảnh đã tải lên")

    # Nút dự đoán
    if st.button("Dự đoán"):
        features = preprocess_and_extract(img)
        features_scaled = scaler.transform([features])
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)

        predicted_label = labels_reverse_dict[prediction[0]]
        confidence = np.max(prediction_proba) * 100

        st.write(f"**Loại trái cây dự đoán:** {predicted_label}")
        st.write(f"**Độ tin cậy:** {confidence:.2f}%")
