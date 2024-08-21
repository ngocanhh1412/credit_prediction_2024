import streamlit as st
import os
import json
import shutil
import pandas as pd
import pickle
from main import train_model

# Load các phiên bản mô hình trong thư mục mlops
def load_model_versions(mlops_dir='D:/DH/Thuc Tap/pj/mlops/'):
    versions = [f for f in os.listdir(mlops_dir) if os.path.isdir(os.path.join(mlops_dir, f)) and f.startswith('Model_v')]
    versions.sort(reverse=True)  # Sắp xếp để lấy phiên bản mới nhất trước
    return versions

# Hiển thị chi tiết một phiên bản mô hình
def display_model_details(version):
    st.header(f"Model Version: {version}")
    version_dir = f'D:/DH/Thuc Tap/pj/mlops/{version}/'

    # Hiển thị các file CSV đã dùng cho quá trình train mô hình
    data_files_path = os.path.join(version_dir, 'data_files.txt')
    if os.path.exists(data_files_path):
        with open(data_files_path, 'r') as f:
            data_files = f.read().splitlines()
        st.subheader("Các file data đã sử dụng:")
        st.write(f"{', '.join(data_files)}")
    else:
        st.warning("Không có thông tin về data files cho mô hình phiên bản này.")

    # Hiển thị ngưỡng điểm từ threshold.txt
    threshold_path = os.path.join(version_dir, 'threshold.txt')
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            threshold = f.read().strip()
        st.subheader("Threshold:")
        st.write(threshold)
    else:
        st.warning("Không có file ngưỡng điểm tín dụng cho mô hình phiên bản này.")

    # Hiển thị các chỉ số metrics với định dạng tùy chỉnh
    metrics_path = os.path.join(version_dir, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        st.subheader("Thông số mô hình:")
        st.write(f"Accuracy: {metrics.get('accuracy'):.4f}")
        st.write(f"Precision: {metrics.get('precision'):.4f}")
        st.write(f"Recall: {metrics.get('recall'):.4f}")
        st.write(f"F1: {metrics.get('f1'):.4f}")
        st.write(f"AUC: {metrics.get('auc'):.4f}")
        st.write(f"Gini: {metrics.get('gini'):.4f}")
    else:
        st.warning("Không có thông số cho mô hình phiên bản này.")

    # Hiển thị các biểu đồ
    # st.subheader("Biểu đồ:")
    confusion_matrix_path = os.path.join(version_dir, 'confusion_matrix.png')
    if os.path.exists(confusion_matrix_path):
        st.image(confusion_matrix_path, caption='Confusion Matrix')
    else:
        st.warning("Không có Confusion Matrix.")

    roc_curve_path = os.path.join(version_dir, 'roc_curve.png')
    if os.path.exists(roc_curve_path):
        st.image(roc_curve_path, caption='ROC Curve')
    else:
        st.warning("Không có ROC Curve.")

    youden_statistic_plot_path = os.path.join(version_dir, 'youden_statistic_plot.png')
    if os.path.exists(youden_statistic_plot_path):
        st.image(youden_statistic_plot_path, caption='Youden\'s J-Statistic Plot')
    else:
        st.warning("Không có Youden's J-Statistic plot.")

# Cập nhật mô hình vào backend
def update_backend_model(version):
    version_dir = f'D:/DH/Thuc Tap/pj/mlops/{version}/'
    backend_dir = 'D:/DH/Thuc Tap/pj/webapp/backend/'

    # Copy model và các file cần thiết vào backend
    os.makedirs(backend_dir, exist_ok=True)
    files_to_copy = ['pipeline_model.pkl', 'scorecard_scores.csv', 'threshold.txt']
    for file in files_to_copy:
        src = os.path.join(version_dir, file)
        dst = os.path.join(backend_dir, file)
        if os.path.exists(src):
            shutil.copy(src, dst)  # Thay thế file cũ
            st.success(f"Thay thế {file} đến backend thành công.")

# Main Dashboard
def main():
    st.title("MLOps Dashboard")

    # Tùy chọn phiên bản mô hình để xem
    versions = load_model_versions()

    if not versions:
        st.warning("Chưa có mô hình nào. Hãy huấn luyện lại mô hình để tạo phiên bản đầu tiên.")
        
        # Tùy chọn retrain mô hình lần đầu tiên
        if st.button("Retrain Model"):
            with st.spinner('Training model...'):
                train_model()
            st.success("Huấn luyện mô hình thành công! Hãy refresh lại trang này để thấy phiên bản mới của mô hình.")
    else:
        selected_version = st.selectbox("Chọn phiên bản mô hình:", versions)

        if selected_version:
            display_model_details(selected_version)

            # Chọn mô hình sử dụng
            if st.button(f"Sử dụng mô hình {selected_version}"):
                update_backend_model(selected_version)
                st.success(f"Mô hình {selected_version} đang được sử dụng!")

        # Tùy chọn retrain mô hình
        if st.button("Retrain Model"):
            with st.spinner('Training model...'):
                train_model()
            st.success("Huấn luyện mô hình thành công! Hãy refresh lại trang này để thấy phiên bản mới của mô hình.")

if __name__ == '__main__':
    main()
