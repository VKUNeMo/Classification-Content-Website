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
                                summary = self.summarizer.summarize(content, top_n=8)
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
            
            st.subheader("Mô tả về label")
            st.write("""
            1. Báo chí, tin tức:
Các trang báo, tạp chí điện tử, kênh truyền thông đưa tin tức, phân tích thời sự, chính trị, xã hội, kinh tế, giải trí,...
- Ví dụ: VnExpress, VietnamNet, BBC, Dân Trí
- Từ khóa gợi ý: “tin mới nhất”, “bản tin”, “phóng sự”, “phân tích thời sự”, “truyền hình trực tuyến”
- Không bao gồm blog cá nhân hoặc forum tự do

2. 18+, shop tình dục, web phim người lớn:
Trang có nội dung nhạy cảm, người lớn, khiêu dâm, hoặc bán sản phẩm phục vụ tình dục.
- Ví dụ: phim18.vip, shopdochoisex.vn
- Từ khóa: “phim người lớn”, “gợi cảm”, “đồ chơi tình dục”, “bao cao su”, “sex toy”, “18+”, “video nóng”, “gái xinh”
- Không áp dụng với nội dung tình cảm thuần túy hoặc y học

3. Cờ bạc, bóng đá, xổ số, lô đề:
Website có nội dung liên quan đến cá cược thể thao, xổ số, đánh bài, lô đề, tài xỉu,...
- Ví dụ: 789bet, xosomienbac.net, kubet
- Từ khóa: “soi kèo”, “tỷ lệ cược”, “xổ số hôm nay”, “nhà cái uy tín”, “chơi lô đề”, “cược bóng đá”, “nổ hũ”, “casino online”
- Không áp dụng với tin tức phân tích thể thao thuần túy không có yếu tố cá độ (=> chuyển về nhãn 1)

4. Vay tín dụng:
Trang cho vay ngân hàng vay tiền mặt, ứng tiền online, tín dụng cá nhân hoặc tài chính tiêu dùng.
- Ví dụ: doctiennhanh.vn, vaytienngay.vn
- Từ khóa: “vay tiền nhanh”, “duyệt vay 15 phút”, “không cần thế chấp”, “lãi suất thấp”, “vay online”

5. Đầu tư tài chính, tiền ảo:
Trang cung cấp thông tin, dịch vụ hoặc quảng bá về đầu tư tài chính, tiền mã hóa, cổ phiếu, mô hình tài chính đa cấp.
- Ví dụ: binance.com, sanforex.com, bitcoinvietnam.news
- Từ khóa: “đầu tư tài chính”, “cổ phiếu”, “mở tài khoản sàn”, “tiền ảo”, “crypto”, “coin”, “blockchain”, “Forex”, “P2P lending”

6. Tổ chức nhà nước, giáo dục, y tế, hành chính:
Website của cơ quan nhà nước, bệnh viện, trường học, tổ chức hành chính hoặc tổ chức công cộng phi lợi nhuận.
- Ví dụ: moet.gov.vn, hanoi.gov.vn, benhvienbachmai.vn
- Từ khóa: “UBND”, “sở y tế”, “trường đại học”, “bệnh viện”, “chính phủ”, “hành chính công”, “thủ tục hành chính”
- Không áp dụng với tin tức có đề cập đến chính phủ (=> chuyển về nhãn 1)

7. E-commerce, shop hợp pháp:
Trang thương mại điện tử hoặc cửa hàng online bán các sản phẩm thông thường, hợp pháp theo luật Việt Nam.
- Ví dụ: shopee.vn, tiki.vn, thegioididong.com
- Từ khóa: “giỏ hàng”, “sản phẩm khuyến mãi”, “đặt mua ngay”, “ship COD”, “giao hàng tận nơi”, “chính sách đổi trả”
- Không áp dụng với các shop chứa sản phẩm cấm hoặc nhạy cảm (chuyển về nhãn 2)

8. Còn lại:
Các trang không thuộc bất kỳ loại nào bên trên.
- Ví dụ: blog cá nhân, forum công nghệ, portfolio cá nhân, trang kỹ thuật chuyên ngành, trang lỗi 404
- Từ khóa: “diễn đàn”, “chia sẻ kinh nghiệm”, “tự học lập trình”, “blog du lịch”, “github page”
- Không chọn nhãn này nếu nội dung có thể phù hợp rõ ràng với 1–7
            """)
