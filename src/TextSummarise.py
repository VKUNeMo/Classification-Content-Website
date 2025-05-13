import os
import numpy as np
import string
import networkx as nx
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from underthesea import word_tokenize, sent_tokenize
import re

class TextSummarizer:
    """Lớp tóm tắt nội dung văn bản sử dụng thuật toán LexRank"""
    
    def __init__(self, word2vec_model_path=None):
        """
        Khởi tạo mô hình tóm tắt LexRank
        
        Args:
            word2vec_model_path (str): Đường dẫn đến mô hình Word2Vec đã được huấn luyện
        """
        self.model = None
        
        # Nếu không cung cấp đường dẫn, tự động tìm mô hình trong thư mục mặc định
        if not word2vec_model_path:
            # Lấy đường dẫn hiện tại của file TextSummarizer.py
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Đi lên một cấp thư mục để tìm thư mục word2vec_model
            project_root = os.path.dirname(current_dir)
            word2vec_model_path = os.path.join(project_root, 'word2vec_model', 'word2vec_model_2.bin')
        
        if os.path.exists(word2vec_model_path):
            self.load_model(word2vec_model_path)
        else:
            print(f"Không tìm thấy mô hình tại đường dẫn: {word2vec_model_path}")
            
    def load_model(self, model_path):
        """
        Tải mô hình Word2Vec từ đường dẫn
        
        Args:
            model_path (str): Đường dẫn đến mô hình Word2Vec
        """
        try:
            self.model = Word2Vec.load(model_path)
            print(f"Đã tải mô hình Word2Vec từ {model_path}")
        except Exception as e:
            print(f"Không thể tải mô hình: {e}")
            self.model = None
# class TextSummarizer:
#     """Lớp tóm tắt nội dung văn bản sử dụng thuật toán LexRank"""
    
#     def __init__(self, word2vec_model_path=''):
#         """
#         Khởi tạo mô hình tóm tắt LexRank
        
#         Args:
#             word2vec_model_path (str): Đường dẫn đến mô hình Word2Vec đã được huấn luyện
#         """
#         self.model = None
#         if word2vec_model_path:
#             self.load_model(word2vec_model_path)
            
#     def load_model(self, model_path):
#         """
#         Tải mô hình Word2Vec từ đường dẫn
        
#         Args:
#             model_path (str): Đường dẫn đến mô hình Word2Vec
#         """
#         try:
#             self.model = Word2Vec.load(model_path)
#             print(f"Đã tải mô hình Word2Vec từ {model_path}")
#         except Exception as e:
#             print(f"Không thể tải mô hình: {e}")
#             self.model = None
  
    def _preprocess_text(self, text):
        """
        Tiền xử lý văn bản: tách câu, tách từ, loại bỏ dấu câu
        
        Args:
            text (str): Văn bản cần xử lý
            
        Returns:
            tuple: (câu gốc, câu đã xử lý)
        """
        text = text.strip()
        lines = text.split("\n")
        sentences = []
        for line in lines:
            line = line.strip()
            if line:
                sentences.extend(sent_tokenize(line))
        # Loại bỏ câu trùng lặp
        unique_sentences = list(dict.fromkeys(sentences))
        
        # Xử lý từng câu
        processed_sentences = []
        for sentence in unique_sentences:
            words = word_tokenize(sentence.lower())
            words = [word for word in words if word not in string.punctuation]
            processed_sentences.append(words)
            
        return unique_sentences, processed_sentences
    
    def _sentence_to_vec(self, sentence):
        """
        Chuyển đổi câu thành vector sử dụng mô hình Word2Vec
        
        Args:
            sentence (list): Danh sách các từ trong câu
            
        Returns:
            numpy.ndarray: Vector đại diện cho câu
        """
        if self.model is None:
            raise ValueError("Cần tải mô hình Word2Vec trước")
            
        word_vectors = [self.model.wv[word] for word in sentence if word in self.model.wv]
        if len(word_vectors) == 0:
            return np.zeros(self.model.vector_size)
        return np.mean(word_vectors, axis=0)
    
    def _get_sentence_vectors(self, processed_sentences):
        """
        Chuyển đổi tất cả câu thành vectors
        
        Args:
            processed_sentences (list): Danh sách các câu đã xử lý
            
        Returns:
            numpy.ndarray: Ma trận các vector câu
        """
        return np.array([self._sentence_to_vec(sentence) for sentence in processed_sentences])
    
    def _apply_lexrank(self, sentence_vectors, sentences, top_n=5):
        """
        Áp dụng thuật toán LexRank để xếp hạng câu
        
        Args:
            sentence_vectors (numpy.ndarray): Ma trận vector câu
            sentences (list): Danh sách các câu gốc
            top_n (int): Số lượng câu cần trích xuất
            
        Returns:
            list: Danh sách câu tóm tắt theo thứ tự xuất hiện ban đầu
        """
        # Tạo ma trận tương đồng
        sim_matrix = cosine_similarity(sentence_vectors)
        
        # Đảm bảo không có giá trị quá nhỏ
        sim_matrix = np.maximum(sim_matrix, 0.05)
        
        # Tạo đồ thị
        graph = nx.from_numpy_array(sim_matrix)
        
        # Áp dụng PageRank để tính điểm cho từng câu
        try:
            scores = nx.pagerank(graph, alpha=0.85, max_iter=100)
        except nx.PowerIterationFailedConvergence:
            # Nếu thuật toán không hội tụ, gán giá trị đều cho mỗi câu
            scores = {i: 1.0 / len(sentences) for i in range(len(sentences))}
        
        # Lấy top_n câu có điểm cao nhất
        ranked_indices = np.argsort([scores[i] for i in range(len(sentences))])[-top_n:][::-1]
        
        # Sắp xếp lại theo thứ tự xuất hiện trong văn bản gốc
        ranked_indices_sorted = sorted(ranked_indices)
        
        # Lấy các câu theo thứ tự đã sắp xếp
        ranked_sentences = [sentences[i] for i in ranked_indices_sorted]
        
        return ranked_sentences
    
    def _clean_text(self, text):
        """
        Làm sạch văn bản, loại bỏ các ký tự không hợp lệ
        
        Args:
            text (str): Văn bản cần làm sạch
            
        Returns:
            str: Văn bản đã làm sạch
        """
        RE_INVALID = re.compile(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]')
        return RE_INVALID.sub("", text)
    
    def summarize(self, text, top_n=None, min_length=None, max_length=None):
        """
        Tóm tắt văn bản sử dụng thuật toán LexRank
        
        Args:
            text (str): Văn bản cần tóm tắt
            top_n (int, optional): Số lượng câu cần trích xuất, nếu None sẽ sử dụng dynamic_top_n
            min_length (int, optional): Không sử dụng trong thuật toán này
            max_length (int, optional): Không sử dụng trong thuật toán này
            
        Returns:
            str: Văn bản đã được tóm tắt
        """
        try:
            if self.model is None:
                return text[:500] + "... (Cần tải mô hình Word2Vec trước khi tóm tắt)"
            
            # Tiền xử lý văn bản
            sentences, processed_sentences = self._preprocess_text(text)
            
            if not sentences:
                return ""
            
            # Tính toán vector cho các câu
            vectors = self._get_sentence_vectors(processed_sentences)
        
            
            # Áp dụng LexRank
            top_sentences = self._apply_lexrank(vectors, sentences, top_n=top_n)
            
            # Tạo bản tóm tắt
            summary = " ".join(top_sentences)
            
            return self._clean_text(summary)
            # return sentences
        except Exception as e:
            # Trả về một phần văn bản gốc nếu có lỗi
            return f"Lỗi khi tóm tắt: {str(e)}\n" + text[:500] + "... (không thể tóm tắt đầy đủ)"
    