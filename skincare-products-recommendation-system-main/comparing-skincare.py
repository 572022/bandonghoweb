# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.manifold import TSNE 
# from scipy.spatial.distance import cdist
# st.markdown("""
#     <style>
#         h1 {
#             color: #e63946;
#             text-align: center;
#         }
#         .stButton > button {
#             background-color: #457b9d;
#             color: white;
#             border-radius: 8px;
#             font-size: 16px;
#             padding: 8px 16px;
#         }
#         .stButton > button:hover {
#             background-color: #1d3557;
#         }
#     </style>
# """, unsafe_allow_html=True)

# st.title("🔍 Tìm kiếm sản phẩm chăm sóc da phù hợp với bạn!")

# st.write("Xin chào! 👋 Nếu bạn đang sử dụng một sản phẩm chăm sóc da yêu thích, tôi có thể giúp bạn tìm các sản phẩm tương tự dựa trên thành phần của chúng. 🧴✨")

# st.write("Vui lòng chọn một sản phẩm bên dưới để tôi có thể gợi ý những sản phẩm tương tự nhé! 😊")
# st.write("📌 Dữ liệu của tôi chứa hơn **1400 sản phẩm**! Tuy nhiên, có thể tôi không có sản phẩm bạn đang tìm. 😔")

# # Load dữ liệu
# # df = pd.read_csv("D:/DELL/Downloads/ML/CK/skincare-products-recommendation-system-main/data/cosmetics.csv")
# df = pd.read_csv("D:\Move_C\skincare-products-recommendation-system-main\data\cosmetics.csv")

# # Chọn danh mục sản phẩm
# category = st.selectbox(label="🗂 Chọn danh mục sản phẩm", options=df['Label'].unique())
# category_subset = df[df['Label'] == category]

# # st.write("Các cột có trong dữ liệu:", df.columns.tolist())

# # Chọn thương hiệu
# brand = st.selectbox(label="🏷 Chọn thương hiệu", options=sorted(category_subset['Brand'].unique()))
# category_brand_subset = category_subset[category_subset['Brand'] == brand]

# # Chọn sản phẩm
# product = st.selectbox(label="🧴 Chọn sản phẩm", options=sorted(category_brand_subset['Name'].unique()))

# ## Hàm hỗ trợ
# # Mã hoá one-hot

# def oh_encoder(tokens):
#     x = np.zeros(N)
#     for ingredient in tokens:
#         idx = ingredient_idx[ingredient]
#         # Dùng từ điển ingredient_idx để biết thành phần nào nằm ở vị trí nào trong vector.

#         x[idx] = 1
#     return x

# def closest_point(point, points):
#     """ Tìm điểm gần nhất trong danh sách các điểm. """
#     return points[cdist([point], points).argmin()]

# if product is not None:
#     category_subset = category_subset.reset_index(drop=True)

#     # Xây dựng từ điển thành phần và ma trận đặc trưng
#     ingredient_idx = {}
#     corpus = []
#     idx = 0
#     for i in range(len(category_subset)):
#         ingredients = category_subset['Ingredients'][i].lower()
#         tokens = ingredients.split(', ')
#         corpus.append(tokens)
#         for ingredient in tokens:
#             if ingredient not in ingredient_idx:
#                 ingredient_idx[ingredient] = idx
#                 idx += 1

#     # Khởi tạo ma trận đặc trưng
#     M = len(category_subset)
#     N = len(ingredient_idx)
#     A = np.zeros((M, N))

#     for i, tokens in enumerate(corpus):
#         A[i, :] = oh_encoder(tokens)
# # Tạo ma trận A kích thước (số sản phẩm x số thành phần).
# # Mỗi dòng là một sản phẩm, mỗi cột là một thành phần.

# # Nút tìm kiếm
# model_run = st.button("🔎 Tìm sản phẩm tương tự")

# if model_run:
#     st.write("📊 Dựa trên thành phần của sản phẩm bạn chọn, đây là **10 sản phẩm tương tự nhất** ✨")

#     # Chạy thuật toán giảm chiều dữ liệu t-SNE
#     model = TSNE(n_components=2, learning_rate=150, random_state=42)
#     tsne_features = model.fit_transform(A)
    
#     # Gán toạ độ 2D vào dataframe
#     category_subset['X'] = tsne_features[:, 0]
#     category_subset['Y'] = tsne_features[:, 1]
    
#     # Lấy toạ độ sản phẩm mục tiêu
#     target = category_subset[category_subset['Name'] == product]
#     target_x, target_y = target['X'].values[0], target['Y'].values[0]
    
#     df1 = pd.DataFrame()
#     df1['point'] = [(x, y) for x, y in zip(category_subset['X'], category_subset['Y'])]
    
#     # Tính khoảng cách Euclid
#     category_subset['distance'] = [cdist(np.array([[target_x, target_y]]), np.array([p]), metric='euclidean') for p in df1['point']]
    
#     # Sắp xếp theo khoảng cách
#     top_matches = category_subset.sort_values(by=['distance'])

#     # Tính số thành phần chung
#     #Đếm số thành phần trùng nhau giữa sản phẩm chọn và các sản phẩm gợi ý.
#     target_ingredients = target.Ingredients.values[0].split(",")
#     target_ingredients = set([x.strip() for x in target_ingredients])
#     top_matches['Thành phần chung'] = [target_ingredients.intersection(set([x.strip() for x in p.split(",")])) for p in top_matches['Ingredients']]

#     # Tỷ giá tiền
#     exchange_rate = 25530

#     # Chuyển đổi giá sang VNĐ và định dạng số
#     top_matches['Price (VND)'] = top_matches['Price'] * exchange_rate
#     top_matches['Price (VND)'] = top_matches['Price (VND)'].apply(lambda x: f"{int(x):,} đ")

#     # Chỉ giữ cột quan trọng
#     top_matches = top_matches[['Label', 'Brand', 'Name', 'Price (VND)', 'Ingredients','Thành phần chung']]
#     top_matches = top_matches.reset_index(drop=True)
#     top_matches = top_matches.drop(top_matches.index[0])  # Bỏ sản phẩm đã chọn

#     st.dataframe(top_matches.head(10))

import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# --- Load dữ liệu & mô hình ---
@st.cache_data
def load_data():
    return pd.read_csv("data/skindataall.csv")

@st.cache_resource
def load_model():
    return joblib.load("logreg_goodstuff.pkl")

df = load_data()
model = load_model()

# --- Tiền xử lý TF-IDF để tính độ tương đồng ---
tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1,2), min_df=1, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Ingredients'].fillna(''))  # đảm bảo không có NaN
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(df.index, index=df['Product']).drop_duplicates()
titles = df[['Product', 'Ingredients', 'Rating_Stars', 'Product_Url']]

# --- Giao diện người dùng ---
st.title("💄 Hệ tư vấn sản phẩm chăm sóc da cá nhân hóa")

col1, col2 = st.columns(2)
with col1:
    skin_tone = st.selectbox("Chọn Skin Tone:", df["Skin_Tone"].dropna().unique())
    eye_color = st.selectbox("Chọn Eye Color:", df["Eye_Color"].dropna().unique())
with col2:
    skin_type = st.selectbox("Chọn Skin Type:", df["Skin_Type"].dropna().unique())
    hair_color = st.selectbox("Chọn Hair Color:", df["Hair_Color"].dropna().unique())

product_input = st.text_input("🔍 Nhập tên sản phẩm bạn muốn tìm sản phẩm tương tự (tuỳ chọn):")

if st.button("🎯 Gợi ý sản phẩm tốt và phù hợp"):
    # --- Lọc sản phẩm theo đặc điểm người dùng ---
    filtered = df[
        (df['Skin_Tone'] == skin_tone) &
        (df['Eye_Color'] == eye_color) &
        (df['Skin_Type'] == skin_type) &
        (df['Hair_Color'] == hair_color)
    ].copy()

    if filtered.empty:
        st.warning("Không tìm thấy sản phẩm phù hợp với thông tin đã chọn.")
    else:
        # --- Dự đoán Good_Stuff bằng mô hình đã huấn luyện ---
        predicted_good = model.predict(filtered['Ingredients'].fillna(''))
        filtered['Predicted_Good'] = predicted_good

        good_products = filtered[filtered['Predicted_Good'] == 1].copy()
        good_products = good_products.sort_values(by="Rating_Stars", ascending=False)

        if good_products.empty:
            st.warning("Không có sản phẩm nào phù hợp được đánh giá tốt.")
        else:
            st.success("✨ Sản phẩm phù hợp với bạn và được mô hình đánh giá tốt:")
            st.dataframe(good_products[["Product", "Brand", "Rating_Stars", "Category", "Product_Url"]].drop_duplicates().head(10))

