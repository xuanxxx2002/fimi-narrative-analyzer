import re
import warnings
import jieba
import jieba.analyse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import umap

warnings.filterwarnings("ignore")

# ── 資料集（模擬 FIMI 貼文）─────────────────────────────────
posts = [
    # 美食主題
    "這家拉麵店的湯頭濃郁鮮甜，每次來都忍不住加麵",
    "週末自己做提拉米蘇，奶油乳酪搭咖啡酒真的絕了",
    "台南的牛肉湯清晨五點就開門，新鮮溫體牛肉超值得",
    "氣炸鍋料理真的太方便，雞翅外酥內嫩完全不油膩",
    "韓式炸雞配啤酒是週五晚上的標配，今天又點了一份",
    "第一次嘗試自製壽司，醋飯比例抓對之後真的很好吃",
    "早午餐推薦這間，班尼迪克蛋的荷蘭醬做得非常細緻",
    "泰式酸辣湯底加了椰奶之後層次豐富，辣度剛剛好",
    # 科技主題
    "新款無線耳機降噪效果驚人，通勤時完全隔絕外界雜音",
    "用 AI 工具自動整理會議記錄，效率提升三倍不誇張",
    "智慧手錶的睡眠追蹤功能讓我發現自己深睡時間嚴重不足",
    "顯示卡價格終於回穩，這個月入手 4K 遊戲終於不卡頓",
    "雲端硬碟自動同步照片，換手機完全不需要手動備份",
    "機械鍵盤打字回饋感太好了，工作效率都跟著提升",
    "電動車充電樁越來越普及，長途旅行的里程焦慮少了很多",
    # 旅遊主題
    "京都嵐山的竹林早晨人少，光線穿透竹葉的感覺太美了",
    "冰島追極光三天終於成功，那片綠色光幕一輩子難忘",
    "花蓮太魯閣步道健行，立霧溪的水色藍到不像真實存在",
    "一個人背包旅行葡萄牙，里斯本的電車和石板路超迷人",
    "沖繩浮潛看到熱帶魚群，海水清澈到五公尺深都看得見",
    "阿里山日出雲海沒有讓人失望，但凌晨三點起床真的冷",
    "首爾弘大周邊街頭小吃太多選擇，一天根本吃不完",
    # 電影與娛樂主題
    "這部科幻片的視覺特效讓人窒息，IMAX 版本完全值回票價",
    "重看千與千尋才發現裡面有好多細節小時候完全沒注意到",
    "新出的推理劇每集結尾都在反轉，害我追到凌晨兩點",
    "現場演唱會的震撼感是串流平台永遠無法替代的體驗",
    "這本小說的世界觀設定太複雜，看了第三遍才搞清楚脈絡",
    "獨立遊戲的劇情設計有時候比 3A 大作更打動人心",
    "podcast 通勤收聽習慣養成之後，感覺每天都多了一小時",
    "這部紀錄片拍攝深海生物，畫面美到讓人忘記在看科學節目",
]

print(f"載入 {len(posts)} 則貼文\n")

# ── 文字前處理 ─────────────────────────────────────────────
STOPWORDS = set([
    "的", "了", "是", "在", "和", "與", "對", "為", "從", "到", "也", "都",
    "而", "及", "或", "但", "被", "把", "讓", "給", "向", "才", "不", "已",
    "將", "會", "要", "有", "人", "個", "這", "那", "就", "上", "中", "下",
    "大", "小", "多", "少", "能", "可", "以", "之", "其", "更", "最", "很",
    "再", "卻", "又", "還", "所", "等", "著", "過", "嗎", "吧", "呢", "啊",
    "一", "不得",
])

def clean(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text)).strip()

def tokenize(text):
    return [w for w in jieba.cut(clean(text), cut_all=False)
            if len(w) >= 2 and w not in STOPWORDS]

# ── 任務 A：TF vs TF-IDF 關鍵字分析 ─────────────────────
TOP_N = 20

# TF
all_words   = [w for post in posts for w in tokenize(post)]
word_counts = Counter(all_words)
top_tf      = word_counts.most_common(TOP_N)

# TF-IDF
def jieba_tokenizer(text):
    return tokenize(text)

tfidf_vec    = TfidfVectorizer(tokenizer=jieba_tokenizer, max_features=500)
tfidf_matrix = tfidf_vec.fit_transform(posts)
tfidf_scores = dict(zip(tfidf_vec.get_feature_names_out(),
                        tfidf_matrix.toarray().mean(axis=0)))
top_tfidf    = sorted(tfidf_scores.items(), key=lambda x: -x[1])[:TOP_N]

print("=== TF Top-10 ===")
for w, c in top_tf[:10]:
    print(f"  {w}: {c}")
print("\n=== TF-IDF Top-10 ===")
for w, s in top_tfidf[:10]:
    print(f"  {w}: {s:.4f}")

# 視覺化
words_tf, counts_tf   = zip(*top_tf)
words_idf, scores_idf = zip(*top_tfidf)

fig_a = make_subplots(rows=1, cols=2,
                      subplot_titles=(f"詞頻 TF Top {TOP_N}", f"TF-IDF Top {TOP_N}"),
                      horizontal_spacing=0.15)
fig_a.add_trace(go.Bar(y=list(words_tf)[::-1], x=list(counts_tf)[::-1],
                       orientation="h", marker_color="steelblue", name="TF"), row=1, col=1)
fig_a.add_trace(go.Bar(y=list(words_idf)[::-1], x=list(scores_idf)[::-1],
                       orientation="h", marker_color="tomato", name="TF-IDF"), row=1, col=2)
fig_a.update_layout(title_text="任務 A：關鍵字分析（TF vs TF-IDF）",
                    height=600, width=1100, showlegend=False)
fig_a.write_html("keyword_analysis.html")
fig_a.show()

# ── 任務 B：Sentence-BERT 向量化 ──────────────────────────
print("\n載入 Sentence-BERT 模型...")
model      = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
embeddings = model.encode(posts, show_progress_bar=True)
print(f"向量矩陣形狀：{embeddings.shape}\n")

# ── 任務 C：K-Means 聚類 + UMAP 視覺化 ───────────────────
N_CLUSTERS = 4
km            = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
cluster_labels = km.fit_predict(embeddings)

# 各聚類 Top 關鍵字
cluster_keywords = {}
for cid in range(N_CLUSTERS):
    idx      = [i for i, c in enumerate(cluster_labels) if c == cid]
    sub_text = [posts[i] for i in idx]
    words    = [w for t in sub_text for w in tokenize(t)]
    top_w    = [w for w, _ in Counter(words).most_common(5)]
    cluster_keywords[cid] = " / ".join(top_w)
    print(f"Cluster {cid}（{len(idx)} 則）: {cluster_keywords[cid]}")

reducer      = umap.UMAP(n_components=2, n_neighbors=5, min_dist=0.3,
                         metric="cosine", random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

df_umap = pd.DataFrame({
    "x":       embedding_2d[:, 0],
    "y":       embedding_2d[:, 1],
    "cluster": [f"Cluster {c}" for c in cluster_labels],
    "preview": [p[:20] + "…" for p in posts],
    "full":    posts,
})
df_umap.to_csv("narrative_clusters.csv", index=False)

fig_c = px.scatter(
    df_umap, x="x", y="y", color="cluster",
    hover_data={"full": True, "x": False, "y": False},
    text="preview",
    title="任務 C：UMAP 語意分布 + K-Means 敘事聚類",
    height=700, width=950,
    template="plotly_white",
)
fig_c.update_traces(marker=dict(size=12, opacity=0.85), textposition="top center",
                    textfont=dict(size=8))
fig_c.write_html("umap_clusters.html")
fig_c.show()

# ── 任務 D：貼文間 Cosine 相似度熱力圖 ───────────────────
sim_matrix = cosine_similarity(embeddings)
labels_short = [f"[{i}] {p[:10]}…" for i, p in enumerate(posts)]

fig_d = go.Figure(data=go.Heatmap(
    z=sim_matrix,
    x=labels_short, y=labels_short,
    colorscale="RdYlGn", zmin=0, zmax=1,
    hovertemplate="x: %{x}<br>y: %{y}<br>similarity: %{z:.3f}<extra></extra>",
))
fig_d.update_layout(
    title="任務 D：貼文語意相似度矩陣（Cosine Similarity）",
    height=800, width=900,
    xaxis=dict(tickfont=dict(size=8)),
    yaxis=dict(tickfont=dict(size=8)),
)
fig_d.write_html("similarity_heatmap.html")
fig_d.show()

print("\n已儲存：keyword_analysis.html、umap_clusters.html、similarity_heatmap.html、narrative_clusters.csv")
