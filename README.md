# 数学質問文分類プログラムの比較検証 (Math Question Classification Evaluation)

本リポジトリは、数学に関する多様な質問記述文を対象に、自然言語処理（NLP）技術と複数の機械学習アルゴリズムを組み合わせ、質問カテゴリの自動分類およびその性能を比較検証するためのエンジニアリング・プロジェクトです。

テキストデータの形態素解析から特徴ベクトル化（TF-IDF）、複数の分類器（SVM, Gaussian NB, k-NN）による学習・評価、ハイパーパラメータの最適化、そして主成分分析（PCA）を用いた特徴空間および決定境界の可視化までの一連のNLPパイプラインを網羅しています。

---

## 1. システムアーキテクチャ・処理フロー

本プロジェクトは、**「特徴量抽出・データセット生成」** と **「モデル学習・比較検証・可視化」** の2つのコンポーネントから構成されています。
```text
[question_list.xlsx] (生データ)
       │
       ▼ (Janome による形態素解析 / わかち書き)
[TfidfVectorizer_Janome.py]
       │
       ├─► Excelへ書き戻し (わかち書きリスト・TF-IDFベクトル)
       ▼
[detaset.csv] (機械学習用データセット: 行=サンプル, 列=特徴量(単語))
       │
       ▼ (ホールドアウト分割 Train:Test = 75:25)
[Compare_MachineLearning.py]
       │
       ├─► 1. 各種モデル(SVM, Gaussian NB, k-NN)の学習・評価
       │      └─► classification_report 出力 (Precision, Recall, F1, Accuracy)
       │
       ├─► 2. k-NN ハイパーパラメータ $k$ の最適化検証
       │      └─► 推移グラフのプロット (knn_result.png)
       │
       └─► 3. 主成分分析 (PCA) による2次元次元削減
              └─► SVM決定境界 (Decision Boundary) とデータ分布の可視化 (PCA_result.png)
```
---

## 2. ディレクトリ構成

```text
.
├── TfidfVectorizer_Janome.py       # テキスト前処理・TF-IDFベクトル化パイプライン
├── Compare_MachineLearning.py       # モデル学習、評価、ハイパーパラメータ最適化・可視化スクリプト
├── question_list.xlsx               # ターゲットとなる質問テキストデータ（Excel形式）
├── detaset.csv                      # ベクトル化完了後の機械学習用データセット（自動生成）
├── knn_result.png                   # k-NNの近傍数 $k$ の変化に伴う Accuracy 推移グラフ（自動生成）
├── PCA_result.png                   # PCAによる2次元射影空間上のSVM決定境界プロット（自動生成）
└── README.md                        # 本説明ドキュメント
```

---

## 3. 環境要件
本プログラムの実行には、Python 3.10 以上の環境および以下の外部ライブラリが必要です。
* データ処理・数値計算: numpy, pandas, openpyxl
* 形態素解析: Janome
* 機械学習・評価: scikit-learn
* 可視化: matplotlib
インストールコマンド
```Bash
pip install numpy pandas matplotlib openpyxl scikit-learn Janome
```

---

## 4. 各スクリプトの詳細と使用方法
### ①特徴量抽出・データセット生成
TfidfVectorizer_Janome.py を実行し、日本語の質問文を機械学習モデルが入力可能な数値ベクトル（特徴量行列）に変換します。
```Bash
python TfidfVectorizer_Janome.py
```
### 処理内部詳細
1. question_list.xlsx より「インデックス」と「質問文」を読み込みます。
2. Janome の Tokenizer を用いて日本語質問文を形態素解析（わかち書き）し、単語単位のスペース区切り文字列を生成します。
3. scikit-learn の TfidfVectorizer を適用し、TF-IDF 特徴量を計算します。なお、token_pattern=r"(?u)\b\w+\b" を指定することで、デフォルトの挙動で除外されがちな「1文字の記号・数式・単語（$x$, $y$, $r$, $\pi$ など）」を確実に特徴量として抽出するよう最適化しています。
4. 抽出された特徴ベクトル、およびわかち書きリストを元のExcelファイルの3列目・4列目にインプレースで書き込み、5列目に全単語リストを記録します。
5. 最終的に、行をサンプル、列を全単語のTF-IDF値としたデータフレームを detaset.csv として出力します。
### ② 機械学習モデルの比較検証・可視化
データセットの準備が完了した後、Compare_MachineLearning.py を実行して分類モデルの構築と検証を行います。
```Bash
python Compare_MachineLearning.py
```
### 処理内部詳細
1. detaset.csv をインポートし、目的変数（label）と説明変数（特徴量ベクトル）に切り分けます。
2. データを訓練セット 75%、テストセット 25% の割合でランダムに分割（random_state=10）します。
3. 以下の3つの分類アルゴリズムをループ処理で一括学習・予測させ、各モデルの適合率（Precision）、再現率（Recall）、F1スコア（F1-score）、正解率（Accuracy）を小数点以下5桁精度でコンソールに出力します。
* Support Vector Machine (SVM): RBFカーネルを採用した非線形分類。
* Gaussian Naive Bayes: 特徴量の条件付き確率にガウス分布を仮定した確率モデル。
* k-Nearest Neighbors (k-NN): 近傍数 $k=16$ を初期値としたインスタンスベースモデル。
4. k-NNハイパーパラメータ検証: 近傍数 $k$ を 1 から 99 まで走査させ、テストデータに対する Accuracy の推移を knn_result.png としてプロット保存します。これにより、過学習（Overfitting）と学習不足（Underfitting）の境界を可視化します。
5. PCAによる特徴空間の可視化: 高次元なTF-IDF特徴量を、情報損失を最小限に抑えつつ主成分分析（PCA）を用いて2次元空間に射影（n_components=2）します。その後、2次元圧縮データ上でSVMモデルを再学習させ、matplotlib.pyplot.contourf を用いて決定境界（Decision Boundary）を描画し、テストサンプルの分布（散布図）と重ね合わせたグラフ PCA_result.png を生成します。

## 参考文献
(1) 岩村陸，岡田浩毅：「Watson によるチャットボットの学習方法」，CIEC 春季カンファレンス論文集，12，pp.136-137（2021）．
(2) 岩村陸，岡田浩毅：「二つの文章に関する類似度スコア計算プログラムの比較検討」，CIEC 春季カンファレンス論文集，14，pp.76-77（2023）．
(3) 小菅李音，高木正則，市川尚：「チャットボットと個別指導を併用した数学教育における理解困難箇所の学習支援の実践と評価」，情報教育シンポジウム論文集，2020，pp.31-38（2020）．
(4) 渥美雅保，村田祐樹，安川葵：「オープンチャットとロボットの連係によるティーチングアシスタントとの協働システム」，人工知能学会全国大会論文集，31（2017）．
(5) IBM：Watson Assistant，https://www.ibm.com/jp-ja/products/watson-assistant ，(2023年10月24日閲覧)．
(6) 中山光樹：『機械学習・深層学習による自然言語処理入門 scikit-learnとTensorFlowを使った実践プログラミング』，pp.2-44，マイナビ出版（2020）．
(7) 同上，pp.14-19．
(8) Python Japan：https://www.python.jp/ ，(2023年10月24日閲覧)．
(9) NumPy：https://numpy.org/ja/ ，(2023年10月24日閲覧)．
(10) pandas：https://pandas.pydata.org/ ，(2023年10月24日閲覧)．
(11) scikit-learn：https://scikit-learn.org/stable/ ，(2023年10月24日閲覧)．
(12) 北研二，津田和彦，獅々堀正幹：『情報検索アルゴリズム』，pp.33-45，共立出版（2002）．
(13) openpyxl：https://openpyxl.readthedocs.io/en/stable/ ，(2023年10月24日閲覧)．
(14) Janome：https://mocobeta.github.io/janome/ ，(2023年10月24日閲覧)．
(15) MathWorks：サポートベクターマシン (SVM)，https://jp.mathworks.com/discovery/support-vector-machine.html ，(2023年10月24日閲覧)．
(16) 古宮嘉那子，伊藤裕佑，佐藤直人，小谷善行：「文書分類のための Negation Naive Bayes」，自然言語処理，20 (2)，pp.161-182（2013）．
(17) IBM：k近傍法 (k-NN)，https://www.ibm.com/jp-ja/topics/knn ，(2023年10月24日閲覧)．
(18) 環境省：https://www.renewable-energy-potential.env.go.jp/RenewableEnergy/dat/report/r03_01/r03_01_chpt3_Part2.pdf ，(2023年10月24日閲覧)．
(19) 関栞，和喜多美月，杉本理：「Watson Assistant によるWeb 型チャットボットの開発 城西大学経営学部におけるチャットボットの実装について」，城西情報科学研究，pp.27-37，城西大学情報科学研究センター（2022）．
(20) 山内一将，川本淳平，堀良彰，櫻井幸一：「機械学習を用いたセッション分類による C&C トラフィック抽出」，暗号と情報セキュリティシンポジウム，p.4C1-5（2014）．
(21) 中山研一朗，正田備也：「機械学習を用いた分類モデルによる冗長な質問文の要点抽出」，第21回情報科学技術フォーラム（FIT2022）．
(22) 東中竜一郎，稲葉通将，水上雅博：『Pythonでつくる対話システム』，オーム社（2020）．
(23) 田中穂積：『自然言語処理 基礎と応用』，電子情報通信学会（1999）．
