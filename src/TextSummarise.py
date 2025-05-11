from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from transformers import AutoModelForSeq2SeqLM
import torch

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
            # Trả về một phần văn bản gốc nếu không thể tóm tắt
            return text[:500] + "... (không thể tóm tắt đầy đủ)"