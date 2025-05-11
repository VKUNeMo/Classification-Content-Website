import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import AutoModelForSeq2SeqLM
import torch
import pandas as pd

class ContentExtractor:
    """Lớp trích xuất nội dung từ URL website"""
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
    
    def extract_content(self, url):
        """
        Trích xuất nội dung từ URL website
        
        Args:
            url (str): URL của website cần trích xuất
            
        Returns:
            str: Nội dung website sau khi được xử lý
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Loại bỏ các thẻ không liên quan
            for tag in soup(['script', 'style', 'header', 'footer', 'nav']):
                tag.decompose()
            
            # Lấy văn bản từ thẻ body
            text = soup.get_text(separator=' ', strip=True)
            
            # Xử lý văn bản
            text = re.sub(r'\s+', ' ', text)  # Loại bỏ khoảng trắng thừa
            text = re.sub(r'\n+', ' ', text)  # Loại bỏ xuống dòng
            
            # Giới hạn độ dài văn bản để tránh lỗi int too big to convert
            if len(text) > 100000:  # Giới hạn ở khoảng 100k ký tự
                text = text[:100000]
            
            return text
        except Exception as e:
            return f"Lỗi khi trích xuất nội dung: {str(e)}"


class TextSummarizer:
    """Lớp tóm tắt nội dung văn bản sử dụng mô hình từ Hugging Face"""
    
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """
        Khởi tạo mô hình tóm tắt
        
        Args:
            model_name (str): Tên mô hình tóm tắt từ Hugging Face
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
    def summarize(self, text, max_length=512, min_length=100):
        """
        Tóm tắt văn bản
        
        Args:
            text (str): Văn bản cần tóm tắt
            max_length (int): Độ dài tối đa của văn bản tóm tắt
            min_length (int): Độ dài tối thiểu của văn bản tóm tắt
            
        Returns:
            str: Văn bản đã được tóm tắt
        """
        try:
            # Giới hạn độ dài đầu vào theo model_max_length của tokenizer
            max_input_length = self.tokenizer.model_max_length
            if max_input_length > max_length:  # Thêm giới hạn bổ sung để tránh lỗi
                max_input_length = max_length
                
            # Tokenize với truncation để đảm bảo không vượt quá max_input_length
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True,
                max_length=max_input_length
            )
            
            inputs = inputs.to(self.device)
            
            # Đảm bảo max_length không quá lớn
            if max_length > max_length:
                max_length = max_length
                
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return summary
        except Exception as e:
            st.error(f"Lỗi khi tóm tắt văn bản: {str(e)}")
            # Trả về một phần văn bản gốc nếu không thể tóm tắt
            return text[:500] + "... (không thể tóm tắt đầy đủ)"
        
class WebsiteClassifier:
    """Lớp phân loại website dựa trên nội dung"""
    
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        """
        Khởi tạo mô hình phân loại
        
        Args:
            model_name (str): Tên mô hình phân loại từ Hugging Face
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        # Định nghĩa các nhãn phân loại (thay đổi theo mô hình cụ thể)
        self.labels = ["Tin tức", "Thương mại điện tử", "Giáo dục", "Giải trí", "Công nghệ"]
        
    def classify(self, text):
        """
        Phân loại website dựa trên nội dung
        
        Args:
            text (str): Nội dung website cần phân loại
            
        Returns:
            dict: Kết quả phân loại với điểm số cho từng nhãn
        """
        # Tạo pipeline phân loại
        classifier = pipeline(
            "text-classification", 
            model=self.model, 
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        
        # Cắt văn bản thành các đoạn để tránh vượt quá giới hạn độ dài
        max_length = self.tokenizer.model_max_length
        chunks = [text[i:i+max_length] for i in range(0, len(text), max_length)]
        
        # Phân loại từng đoạn
        results = []
        for chunk in chunks[:5]:  # Giới hạn số đoạn để tránh tốn thời gian
            result = classifier(chunk)
            results.append(result[0])
        
        # Trả về kết hợp các kết quả (ví dụ lấy phân loại có điểm cao nhất)
        if results:
            # Chuyển đổi kết quả tùy theo mô hình
            scores = {}
            for i, label in enumerate(self.labels):
                scores[label] = sum(result["score"] for result in results if result["label"] == f"LABEL_{i}") / len(results)
            
            return scores
        else:
            return {label: 0.0 for label in self.labels}


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


if __name__ == "__main__":
    app = WebsiteClassifierApp()
    app.run()