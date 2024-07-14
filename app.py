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

def kmeans_color_quantization(image, color_set):
  pixels = image.reshape((-1, 3))
  kmeans = KMeans(n_clusters=len(color_set))
  labels = kmeans.fit_predict(pixels)
  new_colors = kmeans.cluster_centers_

  color_mapping = {}
  used_colors = set()

  for i, center in enumerate(new_colors):
    distances = {color: np.linalg.norm(center - np.array(rgb)) for color, rgb in color_set.items()}
    closest_color = min(distances, key=distances.get)
    color_mapping[i] = np.array(color_set[closest_color])
    used_colors.add(closest_color)

  unused_colors = set(color_set.keys()) - used_colors
  if unused_colors:
    for color in unused_colors:
      least_used_index = min(color_mapping, key=lambda k: np.sum(np.all(image == color_mapping[k], axis=-1)))
      color_mapping[least_used_index] = np.array(color_set[color])

  quantized = np.array([color_mapping[label] for label in labels])
  return quantized.reshape(image.shape)

def apply_dithering(image, color_set):
  h, w, _ = image.shape
  data = image.copy()
  for y in range(h):
    for x in range(w):
      old_pixel = data[y, x].copy()
      distances = {color: np.linalg.norm(old_pixel - np.array(rgb)) for color, rgb in color_set.items()}
      closest_color = min(distances, key=distances.get)
      new_pixel = np.array(color_set[closest_color])
      data[y, x] = new_pixel
      quant_error = old_pixel - new_pixel
      if x + 1 < w:
        data[y, x + 1] = np.clip(data[y, x + 1] + quant_error * 7 / 16, 0, 255)
      if y + 1 < h:
        if x > 0:
          data[y + 1, x - 1] = np.clip(data[y + 1, x - 1] + quant_error * 3 / 16, 0, 255)
        data[y + 1, x] = np.clip(data[y + 1, x] + quant_error * 5 / 16, 0, 255)
        if x + 1 < w:
          data[y + 1, x + 1] = np.clip(data[y + 1, x + 1] + quant_error * 1 / 16, 0, 255)
  return data

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
  image = Image.open(uploaded_file)
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
    quantized_image = kmeans_color_quantization(resized_image, selected_color_set)
    dithered_image = apply_dithering(quantized_image, selected_color_set)
    final_image = Image.fromarray(dithered_image.astype('uint8'), 'RGB')
    st.image(final_image, caption='レゴブロックの設計図', use_column_width=True)
    st.download_button(label='設計図をダウンロード', data=final_image.tobytes(), file_name='lego_design.png', mime='image/png')