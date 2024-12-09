import csv
import os

def search_product_by_code(ma_san_pham):
    # Mở file CSV và đọc dữ liệu
    
    # Create file path dynamically
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_directory, 'resources', 'Data for GUI.csv')

    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        # Duyệt qua từng dòng trong file CSV
        for row in reader:
            if row['ma_san_pham'] == ma_san_pham:
                return row  # Trả về thông tin của sản phẩm nếu mã sản phẩm trùng

    return None  # Nếu không tìm thấy mã sản phẩm trong file

def print_product_info(product_info):
    if product_info:
        print("\nThông tin sản phẩm:")
        print(f"Mã sản phẩm: {product_info['ma_san_pham']}")
        print(f"Tên sản phẩm: {product_info['ten_san_pham']}")
        print(f"Mô tả sản phẩm: {product_info['mo_ta']}")
        print(f"Số sao đánh giá trung bình: {product_info['avg_so_sao']}")
        print(f"Tổng số bình luận tích cực: {product_info['total_positive']}")
        print(f"Tổng số bình luận tiêu cực: {product_info['total_negative']}")
        print(f"Từ khóa tích cực phổ biến nhất: {product_info['most_popular_positive_word']}")
        print(f"Từ khóa tiêu cực phổ biến nhất: {product_info['most_popular_negative_word']}")
    else:
        print("Không tìm thấy sản phẩm với mã này.")

# [Ben] This section only run if the file is run directly (not imported)
if __name__ == "__main__":
    while True:
        # Yêu cầu người dùng nhập mã sản phẩm
        ma_san_pham = input("\nNhập mã sản phẩm: ")
        
        # Tìm sản phẩm theo mã
        product_info = search_product_by_code(ma_san_pham)
        
        # In kết quả
        print_product_info(product_info)
