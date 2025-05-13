from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import AutoModelForSeq2SeqLM
import torch
class WebsiteClassifier:
    """Lớp phân loại website dựa trên nội dung"""
    
    def __init__(self, model_name="phunganhsang/web-content-sumary-cls"):
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
        self.labels = ["Tin tức", "Nội dung người lớn", "Cờ bạc","Vay tín dụng","Đầu tư tài chính","Tổ chức nhà nước","E-commerce", "Chưa xác định"]
        
    def classify(self, text):
        """
        Phân loại website dựa trên nội dung
        
        Args:
            text (str): Nội dung website cần phân loại
            
        Returns:
            dict: Kết quả phân loại với điểm số cho từng nhãn
        """
        # Tạo pipeline phân loại với giới hạn max_length rõ ràng
        classifier = pipeline(
            "text-classification", 
            model=self.model, 
            tokenizer=self.tokenizer,
            device=0 if self.device == "cuda" else -1
        )
        
        # Xác định độ dài tối đa an toàn 
        max_length = min(self.tokenizer.model_max_length, 256)
        
        # Đảm bảo sử dụng văn bản với độ dài phù hợp
        # Chú ý: Cắt text bằng tokenizer để đảm bảo số tokens không vượt quá giới hạn
        encoded = self.tokenizer(text, truncation=True, max_length=max_length)
        truncated_text = self.tokenizer.decode(encoded['input_ids'], skip_special_tokens=True)
        print(f"Truncated text: {truncated_text}")
        try:
            # Phân loại văn bản với max_length được chỉ định rõ ràng
            result = classifier(truncated_text, truncation=True, max_length=max_length)[0]
            
            # Chuyển đổi kết quả thành định dạng phù hợp
            scores = {}
            for i, label in enumerate(self.labels):
                # Kiểm tra xem label có đúng định dạng không
                expected_label = f"LABEL_{i}"
                if result["label"] == expected_label:
                    scores[label] = result["score"]
                else:
                    # Tìm label chính xác trong kết quả nếu có
                    matching_labels = [r for r in result if isinstance(r, dict) and r.get("label") == expected_label]
                    if matching_labels:
                        scores[label] = matching_labels[0]["score"]
                    else:
                        scores[label] = 0.0
            
            return scores
        except Exception as e:
            print(f"Lỗi trong quá trình phân loại: {e}")
            # Trả về điểm số 0 cho tất cả nhãn nếu có lỗi
            return {label: 0.0 for label in self.labels}