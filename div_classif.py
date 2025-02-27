import os
import re
import time
import signal
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
import xgboost as xgb
import matplotlib.pyplot as plt


class HeadlineDivisivenessModel:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()
        self.xgboost_model = None
    
    def get_bert_embeddings(self, sentence):
        inputs = self.tokenizer(sentence, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    
    def load_and_clean_data(self, file_name):
        file_path = os.path.join(self.data_dir, file_name)
        indexes, urls, titles = [], [], []
        
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file if line.strip()]
        
        i = 0
        while i < len(lines):
            if lines[i].isdigit():
                indexes.append(lines[i])
                urls.append(lines[i + 1])
                titles.append(lines[i + 2])
                i += 3
            else:
                i += 1
        
        df = pd.DataFrame({'index': indexes, 'url': urls, 'headline': titles})
        return df[df['headline'].apply(self.contains_valid_characters)].reset_index(drop=True)
    
    @staticmethod
    def contains_valid_characters(text):
        pattern = re.compile(r'^[\u0000-\u007F\u00C0-\u017F\u2010-\u201F\u2013-\u2014\u20AC\u0024\u00A3\u2026\u00BB\u00AB\uE008\uFF08\uFF09\u2009\u200B\u00A0\u0060\u00B4\n]*$')
        return bool(pattern.match(text))
    
    def extract_embeddings(self, df, batch_size=20000, start=180000, end=200000):
        signal.signal(signal.SIGALRM, lambda signum, frame: (_ for _ in ()).throw(Exception("Timeout")))
        
        for j in range(start, end, batch_size):
            batch = df.iloc[j:j+batch_size].copy()
            batch['embedding'] = batch['headline'].apply(self.safe_embedding_extraction)
            batch.to_csv(os.path.join(self.data_dir, f'embeddings_{j}.csv'), index=False)
            print(f"Saved batch {j}")
    
    def safe_embedding_extraction(self, headline):
        try:
            signal.alarm(60)
            return self.get_bert_embeddings(headline)
        except:
            print(f"Timeout or error for: {headline}")
            return None
        finally:
            signal.alarm(0)
    
    def train_model(self, training_file, log_transform=False, normalize=False):
        data = pd.read_csv(os.path.join(self.data_dir, training_file))
        X = np.stack(data['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ')))
        y = data['output_divisiveness_rating'].astype(float)
        
        if log_transform:
            y = np.log1p(y)
        if normalize:
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(X)
            y = (y - y.mean()) / y.std()
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.xgboost_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=800, learning_rate=0.05, max_depth=5)
        self.xgboost_model.fit(X_train, y_train)
        
        self.evaluate_model(X_test, y_test, log_transform)
    
    def evaluate_model(self, X_test, y_test, log_transform):
        y_pred = self.xgboost_model.predict(X_test)
        if log_transform:
            y_test, y_pred = np.expm1(y_test), np.expm1(y_pred)
        
        mse, rmse = np.mean((y_test - y_pred) ** 2), np.sqrt(np.mean((y_test - y_pred) ** 2))
        baseline_rmse = np.sqrt(np.mean((y_test - y_test.mean()) ** 2))
        print(f"RMSE: {rmse}, Baseline RMSE: {baseline_rmse}")
        
        results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        plt.figure(figsize=(8, 6))
        results.boxplot(column=['Predicted'], by='Actual', grid=False)
        plt.xlabel("Actual y values")
        plt.ylabel("Predicted y values")
        plt.title("Predicted vs Actual")
        plt.show()
    
    def predict_new_data(self, input_path, output_path):
        files = [f for f in os.listdir(input_path) if f.startswith("embeddings_")]
        for file in files:
            df = pd.read_csv(os.path.join(input_path, file))
            X = np.stack(df['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' ')))
            df['predicted_divisiveness'] = self.xgboost_model.predict(X)
            df.to_csv(os.path.join(output_path, f"predictions_{file}"), index=False)
            print(f"Saved predictions for {file}")


# Usage
if __name__ == "__main__":
    model = HeadlineDivisivenessModel("/path/to/data")
    df = model.load_and_clean_data("headlines.txt")
    model.extract_embeddings(df)
    model.train_model("training_data.csv")
    model.predict_new_data("/path/to/embeddings", "/path/to/predictions")
