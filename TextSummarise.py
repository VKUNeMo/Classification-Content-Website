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
        
    def summarize(self, text, max_length=1024, min_length=100):
        """
        Tóm tắt văn bản
        
        Args:
            text (str): Văn bản cần tóm tắt
            max_length (int): Độ dài tối đa của văn bản tóm tắt
            min_length (int): Độ dài tối thiểu của văn bản tóm tắt
            
        Returns:
            str: Văn bản đã được tóm tắt
        """
        # Giới hạn độ dài đầu vào
        max_input_length = self.tokenizer.model_max_length
        text = text[:max_input_length]
        
        # Tạo input và thực hiện tóm tắt
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_length)
        inputs = inputs.to(self.device)
        
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