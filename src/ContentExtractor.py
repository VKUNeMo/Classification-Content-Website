from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time

class ContentExtractor:
    """Lớp trích xuất nội dung từ URL website sử dụng Selenium"""
    
    def __init__(self):
        # Cấu hình Chrome options
        chrome_options = Options()
        chrome_options.add_argument('--headless')  # Chạy ở chế độ không giao diện
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        
        # Khởi tạo WebDriver
        self.driver = webdriver.Chrome(options=chrome_options)
    
    def extract_content(self, url):
        """
        Trích xuất nội dung từ URL website với encoding UTF-8
        
        Args:
            url (str): URL của website cần trích xuất
            
        Returns:
            str: Nội dung website sau khi được xử lý
        """
        try:
            # Load trang web
            self.driver.get(url)
            
            # Chờ để đảm bảo JavaScript được tải hoàn toàn
            time.sleep(3)
            
            # Lấy nội dung văn bản từ body
            body = self.driver.find_element(By.TAG_NAME, 'body')
            text = body.text
            
            # Loại bỏ các ký tự không mong muốn và chuẩn hóa
            text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
            
            return text
            
        except Exception as e:
            return f"Lỗi khi trích xuất nội dung: {str(e)}"
        
        finally:
            # Đóng trình duyệt
            self.driver.quit()