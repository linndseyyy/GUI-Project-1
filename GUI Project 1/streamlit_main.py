import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
from product_search_service import search_product_by_code, print_product_info

# 1. Read data
# Change to utf-8 for vietnamese text
data = pd.read_csv("final_data3.csv", encoding='utf-8')
# Load dữ liệu bình luận riêng
comment_data_path = "Danh_gia2.csv"
comment_data = pd.read_csv(comment_data_path, encoding='utf-8')

# Đọc dữ liệu từ file CSV2 (chứa các đánh giá tích cực/tiêu cực)
filtered_path = '/Users/nii/Desktop/GUI Project 1 copy/resources/Data for GUI.csv'
filtered = pd.read_csv(filtered_path)

# 2. Data pre-processing
source = data['noi_dung_binh_luan']
target = data['label']

# Clean NaN values by replacing them with empty string
source = source.fillna('')

text_data = np.array(source)

count = CountVectorizer(max_features=6000)
count.fit(text_data)
bag_of_words = count.transform(text_data)

X = bag_of_words.toarray()

y = np.array(target)

#3. Save models
  
# luu model CountVectorizer (count)
pkl_count = "count_model.pkl"  
with open(pkl_count, 'wb') as file:  
    pickle.dump(count, file)

#4. Load models 
# Đọc model
# import pickle
pkl_filename = "lg_predictor.pkl"  
with open(pkl_filename, 'rb') as file:  
    lg_prediction = pickle.load(file)
# Read count len model
with open(pkl_count, 'rb') as file:  
    count_model = pickle.load(file)

#--------------
# GUI
st.title("Sentiment Analysis Project")
st.write("## Hasaki - Đánh giá tích cực và tiêu cực")

menu = ["Business Objective", "Build Project", "New Prediction", "Product Search for Customers", "Product Search for Business Owners"]
choice = st.sidebar.selectbox('Menu', menu)
st.sidebar.write("""#### Thành viên thực hiện:
                 Lê Gia Linh & Phạm Tường Hân""")
st.sidebar.write("""#### Giảng viên hướng dẫn: Khuất Thuỳ Phương""")
st.sidebar.write("""#### Thời gian thực hiện: 12/2024""")
if choice == 'Business Objective':  
    st.subheader("Giới thiệu doanh nghiệp")  
    st.write("""Hasaki.vn - một hệ thống cửa hàng mỹ phẩm chính hãng và dịch vụ chăm sóc sắc đẹp chuyên sâu với mạng lưới rộng khắp Việt Nam. Với một hệ thống website cho phép khách hàng đặt hàng và để lại bình luận, Hasaki.vn có được một cơ sở dữ liệu khách hàng lớn và đầy tiềm năng khai thác.
    """)  
    st.write(""" Giới hạn hiện tại: Dữ liệu đánh giá là văn bản thô và chưa được xử lý một cách tự động ⇒ Yêu cầu quá trình phân tích thủ công, tốn thời gian và dễ xảy ra sai sót.
    """)  
    st.image("Hasaki.jpg")
    st.subheader("Business Objective")
    st.write("""
    ###### Xây dựng hệ thống/mô hình dự đoán nhằm: 
            1. Phân loại cảm xúc của khách hàng dựa trên các đánh giá (Positive, Neutral, Negative).
        2. Tăng tốc độ và độ chính xác trong việc phản hồi ý kiến của khách hàng.
        3. Hỗ trợ Hasaki.vn và các đối tác cải thiện sản phẩm, dịch vụ, nâng cao sự hài lòng của khách hàng.
    """)  
    st.write("""###### Yêu cầu: Dùng thuật toán Machine Learning algorithms trong Python để phân loại bình luận tích cực, trung tính và tiêu cực.""")
    st.image("Sentiment Analysis.jpg")

elif choice == 'Build Project':
    st.subheader("Build Project")
    st.write("##### 1. Một vài dữ liệu")
    st.dataframe(data[['ma_san_pham','noi_dung_binh_luan', 'label']].head(3))
    st.dataframe(data[['ma_san_pham','noi_dung_binh_luan', 'label']].tail(3))  

    st.write("##### 2. Trực quan hoá Sentiment Analysis")
    st.write("###### Wordcloud bình luận")
    st.image("Wordcloud.png")
    st.write("###### Kiểm tra sự cân bằng dữ liệu")
    st.image("Plot 1.png")
    st.write("""###### ⇒ Dữ liệu không cân bằng, cần thực hiện oversample để cân bằng dữ liệu""")
    st.write("""###### Sau khi thực hiện cân bằng dữ liệu""")
    st.image("Plot 2.png")
   
    st.write("##### 3. Xây dựng mô hình")
    st.write("""Xây dựng một mô hình sử dụng đa dạng các thuật toán gồm Naive Bayes, Logistic Regression và Random Forest. Các mô hình được huấn luyện trên các đánh giá của khách hàng về sản phẩm trên website Hasaki.vn để phân loại thành các mức độ cảm xúc.""")

    st.write("##### 4. Đánh giá")
    st.write("""Xây dựng một mô hình sử dụng đa dạng các thuật toán gồm Naive Bayes, Logistic Regression và Random Forest. Các mô hình được huấn luyện trên các đánh giá của khách hàng về sản phẩm trên website Hasaki.vn để phân loại thành các mức độ cảm xúc.""")
    st.write("""###### Độ chính xác và thời gian chạy model""")
    st.image("Model Performance.png") 
    st.write("""###### Confusion Matrix""")
    st.image("Confusion matrix.png")
    st.write("##### 5.Kết luận: ")
    st.write("###### Mô hình Logistic Regression phù hợp nhất đối với Sentiment Analysis của tập dữ liệu của Hasaki.vn.")

elif choice == 'New Prediction':
    st.subheader("Select data")
    flag = False
    lines = None
    type = st.radio("Upload data or Input data?", options=("Upload", "Input"))
    if type=="Upload":
        # Upload file
        uploaded_file_1 = st.file_uploader("Choose a file", type=['txt', 'csv'])
        if uploaded_file_1 is not None:
            lines = pd.read_csv(uploaded_file_1, header=None)
            st.dataframe(lines)            
            lines = lines[0]     
            flag = True                          
    if type=="Input":        
        content = st.text_area(label="Input your content:")
        if content!="":
            lines = np.array([content])
            flag = True
    
    if flag:
        st.write("Content:")
        if len(lines)>0:
            st.code(lines)        
            x_new = count_model.transform(lines)        
            y_pred_new = lg_prediction.predict(x_new)       
            st.code("New predictions (0: Negative, 1. Neutral, 2: Positive): " + str(y_pred_new))


# Add Product Search for Business Owner
elif choice == "Product Search for Business Owners":
    st.subheader("Product Search for Business Owners")

    # Lựa chọn mã sản phẩm
    product_ids = comment_data['ma_san_pham'].unique()
    selected_product = st.selectbox("Chọn mã sản phẩm", product_ids)
    # Lọc dữ liệu từ CSV1 theo Product ID đã chọn
    filtered_data = comment_data[comment_data['ma_san_pham'] == selected_product]

    # Lọc dữ liệu từ CSV2 theo Product ID tương ứng
    filtered_review_data = filtered[filtered['ma_san_pham'] == selected_product]


    # Kiểm tra và chuyển đổi kiểu dữ liệu ngày tháng nếu cần
    if 'ngay_binh_luan' in comment_data.columns:
        comment_data['ngay_binh_luan'] = pd.to_datetime(comment_data['ngay_binh_luan'], errors='coerce')

        # Kiểm tra xem có dữ liệu NaT nào không
    
    else:
        st.warning("Dữ liệu không có cột ngày tháng.")
        date_range = None

    # Lọc dữ liệu theo mã sản phẩm
    filtered_data = comment_data[comment_data['ma_san_pham'] == selected_product]

    # Bộ chọn thời gian: Chia thành 2 phần - ngày bắt đầu và ngày kết thúc
    if 'ngay_binh_luan' in comment_data.columns:
        st.write("Chọn khoảng thời gian:")
        
        # Chọn ngày bắt đầu
        start_date = st.date_input(
            "Ngày bắt đầu", 
            min_value=comment_data['ngay_binh_luan'].min(),
            max_value=comment_data['ngay_binh_luan'].max()
        )
        
        # Chọn ngày kết thúc
        end_date = st.date_input(
            "Ngày kết thúc", 
            min_value=start_date,  # Ngày kết thúc không được nhỏ hơn ngày bắt đầu
            max_value=comment_data['ngay_binh_luan'].max()
        )
    else:
        st.warning("Dữ liệu không có cột ngày tháng.")
        start_date, end_date = None, None
   # Tổng số bình luận trong khoảng thời gian đã chọn
    # Tổng số bình luận từ start_date đến end_date
    st.write(f'#### Đánh giá tổng quan về sản phẩm {start_date} đến {end_date}')
    total_comments = len(filtered_data)
    st.metric(f"Tổng số bình luận", total_comments)

    # Số sao trung bình từ start_date đến end_date
    average_star_rating = filtered_data['so_sao'].mean()

    # Kiểm tra giá trị NaN cho số sao trung bình
    if not pd.isna(average_star_rating):
        st.metric(f"Số sao trung bình", f"⭐{round(average_star_rating, 2)}/5.0")
    else:
        st.metric(f"Số sao trung bình", "N/A")

    # Hiển thị các bình luận theo số sao
    star_counts = filtered_data['so_sao'].value_counts().sort_index()
    st.bar_chart(star_counts)
    # Biểu đồ cột về số lượng bình luận
    if 'ngay_binh_luan' in filtered_data.columns:
        comment_counts = filtered_data.groupby(filtered_data['ngay_binh_luan'].dt.date).size()
        fig, ax = plt.subplots()
        comment_counts.plot(kind='bar', ax=ax, color='skyblue')
        ax.set_title("Số lượng bình luận theo ngày")
        ax.set_xlabel("Ngày")
        ax.set_ylabel("Số bình luận")
        plt.xticks(rotation=45)
        st.pyplot(fig)
    st.markdown("<hr style='border: 1px solid green;'>", unsafe_allow_html=True)

    # Hiển thị bình luận theo số sao
     # Sử dụng thanh slider để lựa chọn số sao
    st.write(f'#### Hiển thị bình luận về sản phẩm theo số sao từ ngày {start_date} đến {end_date}')
    star_choice = st.slider(
        "Chọn số sao để hiển thị bình luận",
        min_value=1,
        max_value=5,
        value=1,  # Giá trị mặc định
        step=1,
        help="Chọn số sao từ 1 đến 5 để xem bình luận tương ứng"
    )

    # Hiển thị bình luận theo số sao đã chọn
    st.write(f"Bình luận {star_choice} sao")

    # Lọc dữ liệu cho số sao hiện tại
    star_comments = filtered_data[filtered_data['so_sao'] == star_choice][['ma_khach_hang', 'ngay_binh_luan', 'noi_dung_binh_luan']]

    # Đổi tên cột để dễ hiểu
    star_comments = star_comments.rename(columns={
        'ma_khach_hang': 'Mã khách hàng',
        'ngay_binh_luan': 'Thời gian bình luận',
        'noi_dung_binh_luan': 'Bình luận'
    })

    # Mở rộng chiều rộng bảng bình luận (100% chiều rộng của màn hình)
    st.write(star_comments.style.set_table_styles([
        {'selector': 'table', 'props': [('width', '100%')]}  # Đặt bảng chiếm toàn bộ chiều rộng
    ]))

    # Hiển thị bảng các bình luận
    st.dataframe(star_comments)
    st.markdown("<hr style='border: 1px solid green;'>", unsafe_allow_html=True)

    # Hiển thị cụ thể các đánh giá
    st.write(f"#### Đánh giá tích cực và tiêu cực cho sản phẩm")
    positive_negative_counts = [
        int(filtered_review_data['total_positive']),
        int(filtered_review_data['total_negative'])
    ]
    labels = ['Tích cực', 'Tiêu cực']
    colors = ['#2ecc71', '#e74c3c']

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(positive_negative_counts, labels=labels, autopct='%1.1f%%', colors=colors)
    plt.title(f'Phân phối đánh giá cho sản phẩm {selected_product}')
    st.pyplot(fig)

    # Hiển thị các từ khóa phổ biến nhất cho đánh giá tích cực và tiêu cực
    st.write(f"#### Từ khóa phổ biến cho sản phẩm {selected_product}")

    # Hiển thị từ khóa tích cực
    most_popular_positive = filtered_review_data['most_popular_positive_word'].iloc[0]
    st.metric("Từ khóa tích cực phổ biến", most_popular_positive, "✨", delta_color="normal")

    # Hiển thị từ khóa tiêu cực
    most_popular_negative = filtered_review_data['most_popular_negative_word'].iloc[0]
    st.metric("Từ khóa tiêu cực phổ biến", most_popular_negative, "⚠️", delta_color="inverse")


# Added product search functionality
elif choice == 'Product Search for Customers':
    st.subheader("Product Search Customers")
    
    product_code = st.text_input("Enter Product Code:")
    
    if st.button("Tìm kiếm"):
        if product_code:            
            product_info = search_product_by_code(product_code)
            print_product_info(product_info)

            if product_info:
                st.title(f"📦 {product_info['ten_san_pham']}")

                # Product info in a nice box
                st.info(f"📦 {product_info['mo_ta']}")
                
                # Rating with stars
                rating = float(product_info['avg_so_sao'])
                st.markdown(f"### ⭐ Đánh giá: {rating:.1f}/5.0")

                left_col, right_col = st.columns(2)

                # Left column: Pie Chart
                with right_col:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    reviews = [int(product_info['total_positive']), int(product_info['total_negative'])]
                    labels = ['Positive', 'Negative']
                    colors = ['#2ecc71', '#e74c3c']
                    ax.pie(reviews, labels=labels, autopct='%1.1f%%', colors=colors)
                    plt.title('Phân phối đánh giá')
                    st.pyplot(fig)

                # Right column: Review Statistics
                with left_col:
                    with st.expander("📊 Thống kê đánh giá", expanded=True):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Đánh giá tích cực", product_info['total_positive'], "👍", delta_color="normal")
                        with col2:
                            st.metric("Đánh giá tiêu cực", product_info['total_negative'], "👎", delta_color="inverse")


                # Most used words
                with st.expander("📝 Từ được sử dụng nhiều nhất", expanded=True):
                    st.metric("Từ khoá tích cực phổ biến", product_info['most_popular_positive_word'], "✨", delta_color="normal")
                    st.metric("Từ khoá tiêu cực phổ biến", product_info['most_popular_negative_word'], "⚠️", delta_color="inverse")
                  
            else:
                st.error("Không tìm thấy sản phẩm!")
        else:
            st.warning("Hãy nhập mã sản phẩm")