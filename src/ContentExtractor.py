import requests
from bs4 import BeautifulSoup
import re

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
            
            return text
        except Exception as e:
            return f"Lỗi khi trích xuất nội dung: {str(e)}"