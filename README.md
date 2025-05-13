# Classification-Content-Website

## Tạo môi trường ảo
`python -m venv venv`

## Acctive môi trường
### win
`venv\Scripts\activate`
### linux
`source venv/bin/activate`

## Cài thư viện vào môi trường
pip install -r requirements.txt

## List thư viện
pip list

## Nếu có cài thư viện nào mới thì freeze lại vào file
pip freeze > requirements.txt

## Thoát môi trường ảo
deactivate

## Chạy server
streamlit run main.py
