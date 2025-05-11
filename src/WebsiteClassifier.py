from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import AutoModelForSeq2SeqLM
import torch
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
        self.labels = ["Tin tức", "Nội dung người lớn", "Cờ bạc","Vay tín dụng","Đầu tư tài chính","E-commerce","Tổ chức", "Chưa xác định"]
        
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
