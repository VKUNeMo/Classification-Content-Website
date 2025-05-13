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
        Trích xuất nội dung từ URL website với encoding UTF-8
        
        Args:
            url (str): URL của website cần trích xuất
            
        Returns:
            str: Nội dung website sau khi được xử lý
        """
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()

            # Đảm bảo mã hóa là UTF-8
            response.encoding = 'utf-8'

            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Loại bỏ các thẻ không cần thiết
            for tag in soup(['script', 'style', 'header', 'footer', 'nav']):
                tag.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            
            return text
        except Exception as e:
            return f"Lỗi khi trích xuất nội dung: {str(e)}"
