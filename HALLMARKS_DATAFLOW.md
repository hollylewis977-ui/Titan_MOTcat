# MOTCat 50 Hallmarks 数据流全流程详解

本文档详细追踪 **50 个 Hallmark 癌症通路数据**从 CSV 文件到模型前向传播的完整数据流，包括所有关键代码片段。

---

## 总览：数据流图

```
CSV 文件
  │
  ├─→ TITAN Features (PKL)          → slide_to_embedding (768-dim)
  ├─→ RNA Data (CSV)                → rna_df (genes × patients)
  └─→ Hallmarks Signatures (CSV)   → omic_names (50 pathways)
          ↓
    Dataset.__init__()
    - 加载所有数据
    - 创建索引映射
          ↓
    Split.load_split()
    - 读取 train.csv / test.csv
    - 构建 patient_dict
    - 计算 survival bins
          ↓
    Split.apply_scaler()
    - 标准化 RNA 数据
    - 预计算每个通路的列索引
          ↓
    Dataset.__getitem__()
    - WSI: [num_slides, 768]
    - 50 omics: [omic1, ..., omic50]
    - Labels: [label, event_time, c]
          ↓
    collate_MIL_survival_sig_hallmarks()
    - 组装 batch (支持可变数量 omic)
          ↓
    Trainer (hallmarks_trainer.py)
    - 动态 unpack batch
    - 构造 kwargs: {x_path, x_omic1, ..., x_omic50}
          ↓
    Model.forward()
    - Genomic SNNs: 50 pathways → (50, 256)
    - OT Co-attention: WSI ↔ RNA
    - Transformers + Attention Pooling
    - 生存预测 (hazards, S)
          ↓
    Loss & C-index
    - NLL Survival Loss
    - Concordance Index
```

---

## 阶段 1: 数据加载 (Dataset 初始化)

### 1.1 文件位置

**数据目录结构**:
```
data/
├── titan_features/
│   └── TCGA_TITAN_features.pkl       # TITAN 预训练特征
├── data_csvs/
│   └── rna/
│       ├── metadata/
│       │   └── hallmarks_signatures.csv  # 50 个通路定义
│       └── hallmarks/
│           └── <CANCER>/
│               └── rna_clean.csv         # RNA 表达数据
└── splits/
    └── survival/
        └── TCGA_<CANCER>_overall_survival_k=<fold>/
            ├── train.csv
            └── test.csv
```

### 1.2 TITAN 特征加载

**代码位置**: `dataset/dataset_survival.py:45-56`

```python
# 加载 TITAN slide-level 特征 (768-dim)
titan_pkl_path = os.path.join(data_dir, 'titan_features', 'TCGA_TITAN_features.pkl')
with open(titan_pkl_path, 'rb') as f:
    titan_data = pickle.load(f)

# 构建 slide_id → embedding 映射
self.slide_to_embedding = {}
for i, fname in enumerate(titan_data['filenames']):
    # fname 格式: TCGA-06-1087-01Z-00-DX2.1f91f05a-f277-4c98-9955-37e0c83b745f
    self.slide_to_embedding[fname] = titan_data['embeddings'][i]  # 768-dim numpy array
```

**PKL 文件结构**:
```python
{
    'filenames': ['slide_id_1', 'slide_id_2', ...],  # List[str]
    'embeddings': np.array([[768-dim], [768-dim], ...])  # (N_slides, 768)
}
```

### 1.3 RNA 数据加载

**代码位置**: `dataset/dataset_survival.py:58-85`

```python
# 加载 RNA 表达数据
rna_csv_path = os.path.join(data_dir, 'data_csvs', 'rna', 'hallmarks', cancer_type, 'rna_clean.csv')
self.rna_df = pd.read_csv(rna_csv_path)

# CSV 格式:
# sample,       GENE1,  GENE2,  ..., GENEN
# TCGA-XX-XXXX-01A, 5.2,    3.1,    ..., 7.8
# TCGA-YY-YYYY-01A, 4.5,    2.9,    ..., 6.3

# 提取 case_id (前 12 个字符)
self.rna_df['case_id'] = self.rna_df['sample'].str[:12]

# 过滤为主肿瘤样本 (-01)
# TCGA barcode 第 13-15 位: -01=主肿瘤, -11=正常组织
self.rna_df['sample_type'] = self.rna_df['sample'].str[12:15]
self.rna_df = self.rna_df[self.rna_df['sample_type'] == '-01'].copy()

# 去重并建立索引
self.rna_df = self.rna_df.drop_duplicates(subset='case_id', keep='first')
self.rna_df_indexed = self.rna_df.set_index('case_id')
```

### 1.4 Hallmarks 通路定义加载

**代码位置**: `dataset/dataset_survival.py:87-104`

```python
# 加载 50 个 Hallmark 通路的基因集定义
signatures_csv_path = os.path.join(data_dir, 'data_csvs', 'rna', 'metadata', 'hallmarks_signatures.csv')
self.signatures = pd.read_csv(signatures_csv_path)

# CSV 格式 (每列是一个通路):
# HALLMARK_PATHWAY_1,  HALLMARK_PATHWAY_2,  ..., HALLMARK_PATHWAY_50
# GENE_A,              GENE_X,              ..., GENE_M
# GENE_B,              GENE_Y,              ..., GENE_N
# GENE_C,              NaN,                 ..., NaN

# 提取 RNA 数据中的基因列
gene_columns = [c for c in self.rna_df.columns
                if c not in ['Unnamed: 0', 'sample', 'case_id', 'sample_type']]

# 为每个通路筛选有效基因
self.omic_names = []
for col in self.signatures.columns:
    genes = self.signatures[col].dropna().tolist()
    # 只保留在 RNA 数据中存在的基因
    valid_genes = [g for g in genes if g in gene_columns]
    self.omic_names.append(valid_genes)

self.omic_sizes = [len(genes) for genes in self.omic_names]
# 输出: [45, 123, 78, ..., 56]  # 50 个通路的基因数量
```

**关键数据结构**:
```python
self.omic_names = [
    ['GENE1', 'GENE2', 'GENE3'],           # Hallmark 1: 3 个基因
    ['GENE4', 'GENE5', ..., 'GENE50'],     # Hallmark 2: 47 个基因
    ...
    ['GENE200', 'GENE201']                  # Hallmark 50: 2 个基因
]
self.omic_sizes = [3, 47, ..., 2]  # 50 个整数
```

---

## 阶段 2: 划分加载 (Split 构建)

### 2.1 读取 train/test CSV

**代码位置**: `dataset/dataset_survival.py:111-136`

```python
def load_split(self, split_csv_path, external_bins=None):
    # 读取 split CSV
    self.slide_data = pd.read_csv(split_csv_path)

    # CSV 格式:
    # case_id,         slide_id,                             dss_survival_days, dss_censorship
    # TCGA-02-0001,    TCGA-02-0001-01Z-00-DX1.UUID.svs,     1234,              0
    # TCGA-02-0002,    TCGA-02-0002-01Z-00-DX1.UUID.svs,     567,               1

    # 构建患者 → 切片映射
    self.patient_dict = {}
    for idx, row in self.slide_data.iterrows():
        case_id = row['case_id']
        slide_id = row['slide_id']
        # 移除 .svs 后缀以匹配 TITAN filenames
        slide_id_clean = slide_id.rstrip('.svs') if slide_id.endswith('.svs') else slide_id

        if case_id not in self.patient_dict:
            self.patient_dict[case_id] = []
        self.patient_dict[case_id].append(slide_id_clean)

    # 去重得到患者级数据
    self.patients_df = self.slide_data.drop_duplicates(['case_id']).reset_index(drop=True)
```

### 2.2 计算生存时间分箱 (Bins)

**代码位置**: `dataset/dataset_survival.py:138-176`

```python
# 训练集: 计算新的 bins
if external_bins is None:
    # 只用未删失患者计算分位数
    uncensored_df = self.patients_df[self.patients_df['dss_censorship'] < 1]

    # 四分位数离散化
    _, q_bins = pd.qcut(uncensored_df['dss_survival_days'], q=4, retbins=True, labels=False)
    # q_bins: [0, 365, 730, 1095, 5000]  # 示例

    # 扩展边界
    q_bins[-1] = self.patients_df['dss_survival_days'].max() + eps
    q_bins[0] = self.patients_df['dss_survival_days'].min() - eps

# 测试集: 使用训练集的 bins
else:
    q_bins = external_bins

# 应用 bins 到所有患者
disc_labels = pd.cut(
    self.patients_df['dss_survival_days'],
    bins=q_bins,
    labels=False,      # 返回整数标签 0, 1, 2, 3
    right=False,
    include_lowest=True
)

# 处理超出范围的值
disc_labels = disc_labels.fillna(3).astype(int).clip(0, 3)
self.patients_df['disc_label'] = disc_labels

# 构建标签字典 (bin, censorship) → class
self.label_dict = {
    (0, 0): 0, (0, 1): 1,  # Bin 0: 未删失/删失
    (1, 0): 2, (1, 1): 3,  # Bin 1
    (2, 0): 4, (2, 1): 5,  # Bin 2
    (3, 0): 6, (3, 1): 7   # Bin 3
}

# 组合标签
self.patients_df['label'] = self.patients_df.apply(
    lambda row: self.label_dict[(row['disc_label'], int(row['dss_censorship']))],
    axis=1
)
```

---

## 阶段 3: 数据标准化 (Scaler)

### 3.1 计算 Scaler (训练集)

**代码位置**: `dataset/dataset_survival.py:201-225`

```python
def get_scaler(self):
    # 收集所有通路的基因（去重）
    all_genes = []
    for genes in self.omic_names:
        all_genes.extend(genes)
    all_genes = list(set(all_genes))  # 去重

    # 收集训练集所有患者的基因表达
    genomic_data = []
    for idx in range(len(self.patients_df)):
        case_id = self.patients_df.iloc[idx]['case_id']
        if case_id in self.rna_df_indexed.index:
            rna_row = self.rna_df_indexed.loc[case_id]
            genomic_data.append(rna_row[all_genes].values)

    # 拟合 StandardScaler
    genomic_data = np.array(genomic_data)  # (n_patients, n_genes)
    scaler = StandardScaler().fit(genomic_data)

    return (scaler, all_genes)
```

### 3.2 应用 Scaler (训练集/测试集)

**代码位置**: `dataset/dataset_survival.py:227-259`

```python
def apply_scaler(self, scalers):
    scaler, scaler_genes = scalers

    # 构建基因矩阵 (一次性操作，避免 __getitem__ 重复计算)
    genomic_data = []
    for idx in range(len(self.patients_df)):
        case_id = self.patients_df.iloc[idx]['case_id']
        if case_id in self.rna_df_indexed.index:
            rna_row = self.rna_df_indexed.loc[case_id]
            genomic_data.append(rna_row[scaler_genes].values)
        else:
            # 缺失患者用零向量
            genomic_data.append(np.zeros(len(scaler_genes)))

    # 标准化
    genomic_data = np.array(genomic_data, dtype=np.float64)
    genomic_data = scaler.transform(genomic_data)
    self.genomic_matrix = genomic_data.astype(np.float32)  # (n_patients, n_genes)

    # 预计算每个通路的列索引（关键优化！）
    gene_to_col = {g: i for i, g in enumerate(scaler_genes)}
    self.hallmark_col_indices = []
    for genes in self.omic_names:
        valid_indices = [gene_to_col[g] for g in genes if g in gene_to_col]
        if len(valid_indices) > 0:
            self.hallmark_col_indices.append(np.array(valid_indices, dtype=np.int32))
        else:
            self.hallmark_col_indices.append(None)
```

**关键数据结构**:
```python
self.genomic_matrix = np.array([
    [gene1_patient1, gene2_patient1, ..., geneN_patient1],  # 患者 1
    [gene1_patient2, gene2_patient2, ..., geneN_patient2],  # 患者 2
    ...
])  # shape: (n_patients, n_genes)

self.hallmark_col_indices = [
    np.array([0, 5, 12]),          # Hallmark 1: 列索引 [0, 5, 12]
    np.array([3, 7, 8, 15, ...]),  # Hallmark 2: 列索引 [3, 7, 8, ...]
    ...
]  # 50 个 numpy arrays
```

---

## 阶段 4: 样本获取 (__getitem__)

**代码位置**: `dataset/dataset_survival.py:261-295`

```python
def __getitem__(self, idx):
    patient_row = self.patients_df.iloc[idx]
    case_id = patient_row['case_id']
    label = patient_row['disc_label']           # 0-3
    event_time = patient_row['dss_survival_days']
    c = patient_row['dss_censorship']           # 0 or 1

    # === 1. 加载 WSI 特征 ===
    slide_ids = self.patient_dict.get(case_id, [])
    wsi_features = []
    for slide_id in slide_ids:
        if slide_id in self.slide_to_embedding:
            emb = self.slide_to_embedding[slide_id]  # 768-dim numpy
            wsi_features.append(torch.from_numpy(emb).float())

    if len(wsi_features) > 0:
        wsi_features = torch.stack(wsi_features)  # [num_slides, 768]
    else:
        wsi_features = torch.zeros(1, 768)  # Fallback

    # === 2. 加载基因组特征 (50 个通路) ===
    omics = []
    for col_indices in self.hallmark_col_indices:
        if col_indices is not None:
            # 快速 NumPy 高级索引
            omic_values = self.genomic_matrix[idx, col_indices]
            omics.append(torch.from_numpy(omic_values))
        else:
            omics.append(torch.zeros(1))

    # 返回格式: (wsi, omic1, ..., omic50, label, event_time, c, case_id)
    return (wsi_features,) + tuple(omics) + (label, event_time, c, case_id)
```

**返回值示例**:
```python
(
    torch.Tensor([2, 768]),   # WSI: 2 张切片
    torch.Tensor([45]),        # Hallmark 1: 45 个基因
    torch.Tensor([123]),       # Hallmark 2: 123 个基因
    ...
    torch.Tensor([56]),        # Hallmark 50: 56 个基因
    2,                         # label (disc_label)
    1234.0,                    # event_time
    0.0,                       # censorship
    'TCGA-02-0001'            # case_id
)  # 总共 55 个元素 (1 WSI + 50 omics + 4 metadata)
```

---

## 阶段 5: 批处理组装 (Collate Function)

**代码位置**: `utils/utils.py:78-97`

```python
def collate_MIL_survival_sig_hallmarks(batch):
    """
    处理可变数量的 omic 输入
    batch_size=1 时，batch[0] 是单个样本的元组
    """
    item = batch[0]  # (wsi, omic1, ..., omic50, label, event_time, c, case_id)
    n_items = len(item)  # 55

    # WSI
    img = item[0]  # [num_slides, 768]

    # Omics (中间 50 个)
    omics = [item[i].type(torch.FloatTensor) for i in range(1, n_items - 4)]

    # Metadata (最后 4 个)
    label = torch.LongTensor([item[n_items - 4]])       # [1]
    event_time = np.array([item[n_items - 3]])          # [1]
    c = torch.FloatTensor([item[n_items - 2]])          # [1]
    case_id = item[n_items - 1]                         # str

    # 返回列表 (不是元组)
    return [img] + omics + [label, event_time, c, case_id]
```

**DataLoader 调用**:

```python
# utils/utils.py:104-135
def get_split_loader(split_dataset, mode='coattn_hallmarks', batch_size=1, ...):
    if mode == 'coattn_hallmarks':
        collate = collate_MIL_survival_sig_hallmarks

    loader = DataLoader(split_dataset, batch_size=batch_size, collate_fn=collate, ...)
    return loader
```

**Batch 输出格式**:
```python
[
    torch.Tensor([2, 768]),     # [0] WSI
    torch.Tensor([1, 45]),      # [1] omic1
    torch.Tensor([1, 123]),     # [2] omic2
    ...
    torch.Tensor([1, 56]),      # [50] omic50
    torch.LongTensor([1]),      # [51] label
    np.array([1234.0]),         # [52] event_time
    torch.FloatTensor([0.0]),   # [53] c
    'TCGA-02-0001'              # [54] case_id
]  # 55 个元素
```

---

## 阶段 6: 训练循环 (Trainer)

**代码位置**: `trainer/hallmarks_trainer.py:12-91`

```python
def train_loop_survival_hallmarks(epoch, model, loader, optimizer, n_classes, ...):
    model.train()

    for batch_idx, batch_data in enumerate(loader):
        # === 动态解包 ===
        data_WSI = batch_data[0].to(device)          # [num_slides, 768]
        label = batch_data[-4].to(device)            # [1]
        event_time = batch_data[-3]                  # [1]
        c = batch_data[-2].to(device)                # [1]
        # case_id = batch_data[-1]                   # 训练时不用

        # 中间所有元素都是 omic
        n_omics = len(batch_data) - 5  # 55 - 5 = 50
        omics = [batch_data[i].type(torch.FloatTensor).to(device)
                 for i in range(1, n_omics + 1)]

        # === 构建模型输入 (关键！) ===
        kwargs = {'x_path': data_WSI}
        for i, omic in enumerate(omics):
            kwargs[f'x_omic{i+1}'] = omic  # x_omic1, x_omic2, ..., x_omic50

        # === 前向传播 ===
        hazards, S, Y_hat, A = model(**kwargs)

        # === 损失计算 ===
        loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
        risk = -torch.sum(S, dim=1).detach().cpu().numpy()

        # === 反向传播 ===
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # === 计算 C-index ===
    c_index_train = concordance_index_censored(
        (1 - all_censorships).astype(bool),
        all_event_times,
        all_risk_scores
    )[0]
```

**kwargs 内容示例**:
```python
kwargs = {
    'x_path': torch.Tensor([2, 768]),      # WSI
    'x_omic1': torch.Tensor([1, 45]),      # Hallmark 1
    'x_omic2': torch.Tensor([1, 123]),     # Hallmark 2
    ...
    'x_omic50': torch.Tensor([1, 56])      # Hallmark 50
}
```

---

## 阶段 7: 模型前向传播 (MOTCat)

**代码位置**: `models/model_motcat.py:174-200`

```python
def forward(self, **kwargs):
    x_path = kwargs['x_path']  # (N_slide, 768) e.g., [2, 768]

    # === 提取所有 omic 输入 ===
    x_omic = [kwargs['x_omic%d' % i] for i in range(1, len(self.omic_sizes)+1)]
    # x_omic[0]: [1, 45]   (Hallmark 1)
    # x_omic[1]: [1, 123]  (Hallmark 2)
    # ...
    # x_omic[49]: [1, 56]  (Hallmark 50)

    # === Step 1: WSI 投影到 256 维 ===
    X_256 = self.proj_ot(x_path)  # [2, 768] → [2, 256]

    # === Step 2: 基因组 SNN 处理 (50 个独立网络) ===
    h_omic_list = [self.sig_networks[idx](sig_feat)
                   for idx, sig_feat in enumerate(x_omic)]
    # h_omic_list[0]: [1, 256]  (Hallmark 1 embedding)
    # h_omic_list[1]: [1, 256]  (Hallmark 2 embedding)
    # ...

    G_256 = torch.stack(h_omic_list)  # [50, 256]
```

### 7.1 最优传输 Co-Attention

**代码位置**: `models/model_motcat.py:188-197`

```python
    # === Step 3: OT 计算传输矩阵 ===
    X_256_ot = X_256.unsqueeze(1)  # [2, 256] → [2, 1, 256]
    G_256_ot = G_256.unsqueeze(1)  # [50, 256] → [50, 1, 256]

    A_coattn, ot_dist = self.coattn(X_256_ot, G_256_ot)
    # A_coattn: [1, 1, 50, 2]  (传输矩阵，已转置)

    P = A_coattn.squeeze(0).squeeze(0).T  # [2, 50]
    # P[i, j] 表示第 i 个 patch 与第 j 个通路的传输量
```

### 7.2 FiLM 调制

**代码位置**: `models/model_motcat.py:198-220`

```python
    # === Step 4: 计算 RNA 上下文 ===
    # 用传输矩阵加权平均 RNA embeddings
    P_norm = P / (P.sum(dim=1, keepdim=True) + eps)  # [2, 50]
    rna_context = torch.matmul(P_norm, G_256)  # [2, 50] @ [50, 256] = [2, 256]

    # === Step 5: FiLM 调制 ===
    gamma, beta = self.film_generator(rna_context)
    # gamma: [2, 256]
    # beta:  [2, 256]

    X_256_modulated = gamma * X_256 + beta  # [2, 256]
    # WSI 特征被 RNA 上下文调制
```

### 7.3 Transformer 聚合

**代码位置**: `models/model_motcat.py:222-250`

```python
    # === Step 6: Path Transformer ===
    X_256_trans = self.path_transformer(X_256_modulated.unsqueeze(0))  # [1, 2, 256]
    X_256_trans = X_256_trans.squeeze(0)  # [2, 256]

    # Attention Pooling
    A_path, h_path = self.path_attention_head(X_256_trans)
    # A_path: [2, 1] (attention weights)
    # h_path: [2, 256]

    A_path = torch.transpose(A_path, 1, 0)  # [1, 2]
    A_path = F.softmax(A_path, dim=1)       # Normalize
    h_path = torch.mm(A_path, h_path)       # [1, 2] @ [2, 256] = [1, 256]
    h_path = self.path_rho(h_path).squeeze() # [256]

    # === Step 7: Omic Transformer ===
    G_256_trans = self.omic_transformer(G_256.unsqueeze(0))  # [1, 50, 256]
    G_256_trans = G_256_trans.squeeze(0)  # [50, 256]

    A_omic, h_omic = self.omic_attention_head(G_256_trans)
    A_omic = torch.transpose(A_omic, 1, 0)
    A_omic = F.softmax(A_omic, dim=1)
    h_omic = torch.mm(A_omic, h_omic)    # [1, 50] @ [50, 256] = [1, 256]
    h_omic = self.omic_rho(h_omic).squeeze()  # [256]
```

### 7.4 多模态融合 + 分类

**代码位置**: `models/model_motcat.py:252-270`

```python
    # === Step 8: Fusion ===
    if self.fusion == 'concat':
        h = self.mm(torch.cat([h_path, h_omic]))  # [512] → [256]
    elif self.fusion == 'bilinear':
        h = self.mm(h_path.unsqueeze(0), h_omic.unsqueeze(0)).squeeze()

    # === Step 9: 生存预测 ===
    logits = self.classifier(h)  # [256] → [4] (n_classes=4 bins)
    Y_hat = torch.topk(logits, 1, dim=0)[1]

    # 计算风险函数
    hazards = torch.sigmoid(logits)  # [4]
    S = torch.cumprod(1 - hazards, dim=0)  # Survival function

    return hazards, S, Y_hat, A_coattn
```

**返回值**:
```python
hazards: [4]       # 每个 bin 的风险概率
S: [4]             # 累积生存概率
Y_hat: [1]         # 预测的 bin
A_coattn: [1,1,50,2]  # OT 传输矩阵 (用于可解释性)
```

---

## 阶段 8: 损失计算与优化

**代码位置**: `utils/utils.py:249-265`

```python
def nll_loss(hazards, S, Y, c, alpha=0.4, eps=1e-7):
    """
    Negative Log-Likelihood Survival Loss

    Args:
        hazards: [batch, n_bins]  # 风险函数
        S: [batch, n_bins]        # 生存函数
        Y: [batch]                # 真实 bin (0-3)
        c: [batch]                # 删失状态 (0=未删失, 1=删失)
    """
    batch_size = len(Y)
    Y = Y.view(batch_size, 1)
    c = c.view(batch_size, 1).float()

    # S(-1) = 1 (所有人在时间 0 时存活)
    S_padded = torch.cat([torch.ones_like(c), S], 1)

    # 未删失损失: -log[S(t-1)] - log[h(t)]
    uncensored_loss = -(1 - c) * (
        torch.log(torch.gather(S_padded, 1, Y).clamp(min=eps)) +
        torch.log(torch.gather(hazards, 1, Y).clamp(min=eps))
    )

    # 删失损失: -log[S(t)]
    censored_loss = -c * torch.log(
        torch.gather(S_padded, 1, Y+1).clamp(min=eps)
    )

    # 组合损失 (加权未删失样本)
    loss = (1 - alpha) * (censored_loss + uncensored_loss) + alpha * uncensored_loss
    return loss.mean()
```

---

## 阶段 9: C-Index 评估

**代码位置**: `trainer/hallmarks_trainer.py:94-160`

```python
def validate_survival_hallmarks(cur, epoch, model, loader, ...):
    model.eval()
    all_risk_scores = []
    all_censorships = []
    all_event_times = []

    for batch_idx, batch_data in enumerate(loader):
        # ... [解包数据] ...

        with torch.no_grad():
            hazards, S, Y_hat, A = model(**kwargs)

        # 计算风险分数 (负生存概率总和)
        risk = -torch.sum(S, dim=1).cpu().numpy()

        all_risk_scores.append(risk)
        all_censorships.append(c.cpu().numpy())
        all_event_times.append(event_time)

    # 计算 Concordance Index
    from sksurv.metrics import concordance_index_censored
    c_index = concordance_index_censored(
        event_indicator=(1 - np.array(all_censorships)).astype(bool),
        event_time=np.array(all_event_times),
        estimate=np.array(all_risk_scores),
        tied_tol=1e-08
    )[0]

    return patient_results, c_index
```

---

## 关键设计亮点

### 1. 预计算列索引（性能优化）

**问题**: 每次 `__getitem__` 都对 50 个通路做字符串查找会非常慢。

**解决**: 在 `apply_scaler()` 时预计算列索引。

```python
# 慢速方案 (每次查找字符串)
omic_values = self.rna_df.loc[case_id, pathway_genes].values

# 快速方案 (整数索引)
col_indices = self.hallmark_col_indices[pathway_idx]  # np.array([3, 7, 15, ...])
omic_values = self.genomic_matrix[patient_idx, col_indices]
```

### 2. 动态 Kwargs 构造（灵活性）

**问题**: 模型需要接收可变数量的 omic 输入 (50 个)。

**解决**: Trainer 动态构造 `kwargs` 字典。

```python
# 灵活支持任意数量 omic
kwargs = {'x_path': data_WSI}
for i, omic in enumerate(omics):
    kwargs[f'x_omic{i+1}'] = omic

hazards, S, Y_hat, A = model(**kwargs)
```

### 3. 最优传输矩阵语义

**OT 矩阵 `P`**:
- `P[i, j]`: 第 i 个 WSI patch 与第 j 个 Hallmark 通路的关联强度
- 用于加权聚合 RNA 上下文
- 归一化后用作注意力权重

### 4. Bin 对齐策略

**训练集**: 从未删失样本计算四分位数。

**测试集**: 使用训练集的 bins，超出范围的值 clip 到边界。

```python
# 训练集
train_split = Generic_MIL_Survival_Split_Hallmarks(dataset, train_csv, external_bins=None)

# 测试集 (复用训练 bins)
test_split = Generic_MIL_Survival_Split_Hallmarks(dataset, test_csv, external_bins=train_split.bins)
```

---

## 数据流总结表

| 阶段 | 输入 | 输出 | 关键操作 |
|------|------|------|----------|
| **1. Dataset 初始化** | PKL, CSV | `slide_to_embedding`, `rna_df`, `omic_names` | 加载文件、构建映射 |
| **2. Split 加载** | train.csv | `patient_dict`, `bins`, `patients_df` | 患者聚合、分箱 |
| **3. 标准化** | RNA 矩阵 | `genomic_matrix`, `hallmark_col_indices` | StandardScaler、预计算索引 |
| **4. __getitem__** | patient idx | WSI [N,768] + 50 omics | NumPy 索引、Tensor 转换 |
| **5. Collate** | 单样本元组 | Batch 列表 | 动态解包、类型转换 |
| **6. Trainer** | Batch 列表 | kwargs 字典 | 动态构造 x_omic1..x_omic50 |
| **7. Model** | kwargs | hazards [4], S [4] | SNN → OT → FiLM → Transformer → Classifier |
| **8. Loss** | hazards, S, Y, c | loss (scalar) | NLL Survival Loss |
| **9. C-Index** | risk, event_time, c | c_index (scalar) | Concordance Index |

---

## 完整示例：单样本追踪

### 输入
```
case_id: TCGA-02-0001
slide_id: TCGA-02-0001-01Z-00-DX1.UUID (有 2 张切片)
dss_survival_days: 1234
dss_censorship: 0 (未删失)
```

### 流程
1. **Dataset**: 查找 2 张切片的 TITAN 特征 → `[2, 768]`
2. **Dataset**: 查找患者的 RNA 数据 → `genomic_matrix[patient_idx, :]`
3. **Dataset**: 按 50 个通路索引切分 → 50 个 tensors
4. **Collate**: 组装成 `[WSI] + [omic1..omic50] + [label, time, c, id]`
5. **Trainer**: 构造 `kwargs = {x_path, x_omic1, ..., x_omic50}`
6. **Model**:
   - WSI `[2, 768]` → proj → `[2, 256]`
   - 50 omics → 50 SNNs → `[50, 256]`
   - OT → `P [2, 50]`
   - FiLM → `X_modulated [2, 256]`
   - Transformer → `h_path [256]`, `h_omic [256]`
   - Concat → `h [512]` → FC → `h [256]`
   - Classifier → `logits [4]` → `hazards [4]`, `S [4]`
7. **Loss**: `nll_loss(hazards, S, label=2, c=0)`
8. **C-Index**: `risk = -sum(S) = -2.5` (示例值)

---

## 相关文件清单

| 文件 | 作用 |
|------|------|
| `main_hallmarks.py` | 入口脚本，训练循环 |
| `dataset/dataset_survival.py` | Dataset 和 Split 类 |
| `utils/utils.py` | Collate 函数、DataLoader |
| `trainer/hallmarks_trainer.py` | 训练和验证循环 |
| `models/model_motcat.py` | MOTCat 模型定义 |
| `models/model_coattn.py` | MCAT 基线模型 |

---

## 常见问题

### Q1: 为什么 __getitem__ 返回 55 个元素？
**A**: 1 个 WSI + 50 个 omics + 4 个 metadata (label, event_time, c, case_id)

### Q2: 为什么用 `genomic_matrix[idx, col_indices]` 而不是 DataFrame？
**A**: NumPy 高级索引比 pandas `.loc` 快 10-100 倍，避免每次 `__getitem__` 重复查找。

### Q3: 测试集的 survival bins 怎么算？
**A**: 使用训练集的 bins (`external_bins=train_split.bins`)，确保标签语义一致。

### Q4: FiLM 调制的作用是什么？
**A**: 让 RNA 通路信息动态调节 WSI 特征，实现 feature-wise 的跨模态融合。

### Q5: 为什么 risk = -sum(S)？
**A**: S 是生存概率，越低表示风险越高。取负号后，risk 越高 → 生存概率越低 → 预后越差。

---

## 参考资料

- **论文**: [Multimodal Optimal Transport-based Co-Attention Transformer](https://arxiv.org/abs/2306.08330)
- **MCAT**: [原始多模态 co-attention 实现](https://github.com/mahmoodlab/MCAT)
- **POT**: [Python Optimal Transport 库](https://github.com/PythonOT/POT)
