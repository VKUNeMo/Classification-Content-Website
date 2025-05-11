import streamlit as st
import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ContentExtractor import ContentExtractor
from src.TextSummarise import TextSummarizer
from src.WebsiteClassifier import WebsiteClassifier
import pandas as pd
class WebsiteClassifierApp:
    """Lớp ứng dụng Streamlit để phân loại website"""
    
    def __init__(self):
        """Khởi tạo ứng dụng"""
        self.content_extractor = ContentExtractor()
        self.summarizer = None
        self.classifier = None
        
        # Thiết lập trang Streamlit
        st.set_page_config(
            page_title="Website Classifier",
            page_icon="🌐",
            layout="wide"
        )
        
    def load_models(self):
        """Tải các mô hình khi được yêu cầu"""
        with st.spinner("Đang tải mô hình tóm tắt..."):
            self.summarizer = TextSummarizer()
        
        with st.spinner("Đang tải mô hình phân loại..."):
            self.classifier = WebsiteClassifier()
    
    def run(self):
        """Chạy ứng dụng Streamlit"""
        st.title("Website Classifier 🌐")
        st.write("Ứng dụng phân loại website dựa trên nội dung")
        
        # Tạo các tab cho ứng dụng
        tab1, tab2 = st.tabs(["Phân loại URL", "Thông tin"])
        
        with tab1:
            url = st.text_input("Nhập địa chỉ URL website cần phân loại:")
            
            col1, col2 = st.columns(2)
            with col1:
                summarize = st.checkbox("Tóm tắt nội dung trước khi phân loại", value=True)
            with col2:
                show_content = st.checkbox("Hiển thị nội dung website", value=False)
            
            if st.button("Phân loại"):
                if not url:
                    st.error("Vui lòng nhập URL website")
                else:
                    if not self.summarizer or not self.classifier:
                        self.load_models()
                    
                    # Trích xuất nội dung
                    with st.spinner("Đang trích xuất nội dung từ website..."):
                        content = self.content_extractor.extract_content(url)
                    
                    if content.startswith("Lỗi"):
                        st.error(content)
                    else:
                        # Tóm tắt nội dung nếu cần
                        if summarize:
                            with st.spinner("Đang tóm tắt nội dung..."):
                                summary = self.summarizer.summarize(content)
                                st.success("Đã tóm tắt nội dung thành công!")
                                
                                if show_content:
                                    st.subheader("Nội dung tóm tắt:")
                                    st.write(summary)
                            
                            # Phân loại nội dung tóm tắt
                            with st.spinner("Đang phân loại website..."):
                                classification_results = self.classifier.classify(summary)
                        else:
                            # Phân loại nội dung gốc
                            with st.spinner("Đang phân loại website..."):
                                classification_results = self.classifier.classify(content)
                                
                                if show_content:
                                    st.subheader("Nội dung website:")
                                    st.write(content[:2000] + "..." if len(content) > 2000 else content)
                        
                        # Hiển thị kết quả phân loại
                        st.subheader("Kết quả phân loại:")
                        
                        # Chuyển đổi kết quả thành DataFrame để hiển thị
                        results_df = pd.DataFrame({
                            'Loại website': list(classification_results.keys()),
                            'Điểm số': list(classification_results.values())
                        })
                        
                        results_df = results_df.sort_values('Điểm số', ascending=False)
                        
                        # Hiển thị biểu đồ
                        st.bar_chart(results_df.set_index('Loại website'))
                        
                        # Hiển thị phân loại có điểm cao nhất
                        st.success(f"Website được phân loại là: **{results_df.iloc[0]['Loại website']}** với điểm số: {results_df.iloc[0]['Điểm số']:.4f}")
        
        with tab2:
            st.header("Thông tin về ứng dụng")
            st.write("""
            Ứng dụng này sử dụng các mô hình học máy để phân loại website dựa trên nội dung:
            
            1. **Trích xuất nội dung**: Ứng dụng trích xuất nội dung văn bản từ URL website.
            2. **Tóm tắt nội dung**: Sử dụng mô hình BART từ Facebook để tóm tắt nội dung website.
            3. **Phân loại website**: Sử dụng mô hình phân loại văn bản để xác định danh mục của website.
            
            Ứng dụng được phát triển sử dụng Streamlit và Hugging Face Transformers.
            """)
            
            st.subheader("Hướng dẫn sử dụng")
            st.write("""
            1. Nhập URL website cần phân loại.
            2. Chọn tùy chọn tóm tắt nội dung nếu muốn.
            3. Nhấn nút "Phân loại" để bắt đầu quá trình.
            4. Xem kết quả phân loại website.
            """)
