import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.WebsitrClassifierApp import WebsiteClassifierApp 

if __name__ == "__main__":
    app = WebsiteClassifierApp()
    app.run()