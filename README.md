# fimi-narrative-analyzer

針對 FIMI（外國資訊操控與干預）貼文進行中文 NLP 分析，透過關鍵字萃取、語意嵌入、聚類與相似度矩陣，識別重複性敘事主題。

## 分析流程

```
模擬 FIMI 貼文（30 則）
         ↓
  jieba 斷詞 + 停用詞過濾
         ↓
┌────────────────────────────────────────────┐
│  任務 A          任務 B        任務 C / D   │
│  TF vs TF-IDF   Sentence-BERT  UMAP 降維   │
│  關鍵字排名      向量化         K-Means 聚類 │
│                               相似度矩陣   │
└────────────────────────────────────────────┘
```

## 輸出

| 檔案 | 內容 |
|---|---|
| `keyword_analysis.html` | TF vs TF-IDF Top-20 關鍵字對照（互動式）|
| `umap_clusters.html` | UMAP 2D 語意分布 + K-Means 敘事聚類（互動式）|
| `similarity_heatmap.html` | 30×30 貼文語意相似度矩陣（互動式）|
| `narrative_clusters.csv` | 每則貼文的 UMAP 座標、聚類標籤與內容 |

## 快速開始

**安裝依賴**

```bash
python -m pip install jieba sentence-transformers umap-learn plotly pandas scikit-learn
```

**執行**

```bash
python fimi_narrative_analyzer.py
```

## 替換自己的資料

將程式碼中的 `posts` 列表替換為：

```python
df = pd.read_csv("your_file.csv")
posts = df["text"].tolist()
```

## 參考

- [FIMI — European External Action Service](https://www.eeas.europa.eu/eeas/tackling-disinformation_en)
- [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
