import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_cropper import st_cropper
import cv2
from sklearn.cluster import KMeans

COLOR_SETS = {
  "赤, 青, 黄": {
    'red': [255, 0, 0],
    'blue': [0, 0, 255],
    'yellow': [255, 255, 0]
  },
  "5色": {
    'yellow': [255, 255, 0],
    'white': [255, 255, 255],
    'light_gray': [192, 192, 192],
    'dark_gray': [96, 96, 96],
    'black': [0, 0, 0]
  }
}

def remove_background(image):
  if image.shape[2] == 4:
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)  # RGBA to RGB
  mask = np.zeros(image.shape[:2], np.uint8)
  bgd_model = np.zeros((1, 65), np.float64)
  fgd_model = np.zeros((1, 65), np.float64)
  rect = (1, 1, image.shape[1]-1, image.shape[0]-1)
  cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
  mask2 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
  image = image * mask2[:, :, np.newaxis]
  return image

def resize_image(image, size):
  return cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)

def replace_colors(image, color_set):
  original_shape = image.shape
  image = image.reshape((-1, 3))
  kmeans = KMeans(n_clusters=len(color_set))
  kmeans.fit(image)
  centroids = kmeans.cluster_centers_
  labels = kmeans.labels_

  new_colors = np.array(list(color_set.values()))
  label_color_map = {i: new_colors[i] for i in range(len(new_colors))}
  replaced_image = np.array([label_color_map[label] for label in labels])
  replaced_image = replaced_image.reshape(original_shape)

  return replaced_image.astype(np.uint8)

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
  image = Image.open(uploaded_file)
  image = ImageOps.exif_transpose(image)  # 画像の向きを修正
  st.image(image, caption='アップロードされた画像', use_column_width=True)

  cropped_image = st_cropper(image, aspect_ratio=(1, 1))
  st.image(cropped_image, caption='トリミングされた画像', use_column_width=True)

  size_option = st.selectbox("ドット絵のサイズを選択してください", options=["48x48", "96x96"])
  size = (48, 48) if size_option == "48x48" else (96, 96)

  color_set_name = st.selectbox("変換後の色を選択してください", options=list(COLOR_SETS.keys()))
  selected_color_set = COLOR_SETS[color_set_name]

  if st.button("レゴブロックの設計図を生成"):
    cropped_image = np.array(cropped_image)
    bg_removed_image = remove_background(cropped_image)
    resized_image = resize_image(bg_removed_image, size)
    
    replaced_image = replace_colors(resized_image, selected_color_set)
    
    final_image = Image.fromarray(replaced_image.astype('uint8'), 'RGB')
    st.image(final_image, caption='レゴブロックの設計図', use_column_width=True)
    st.download_button(label='設計図をダウンロード', data=final_image.tobytes(), file_name='lego_design.png', mime='image/png')
