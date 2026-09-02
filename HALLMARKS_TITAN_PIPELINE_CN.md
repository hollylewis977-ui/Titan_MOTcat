# MOTCat 新数据集(`data/`) + TITAN + Hallmarks：无验证集训练 & 最后 epoch Test C-index（对比 `MOTCat-main.zip`）

这份文档解释：为了让 MOTCat 在你新增的 `data/` 数据集上跑通，并满足“**不使用验证集**、**只用最后一个 epoch 在 test 上算 C-index**”的实验设定，当前仓库相对原版 `MOTCat-main.zip` 具体改了哪些代码、每个改动为什么必须做。

---

## 1. 你的实验需求（需求 → 代码落点）

你的需求可以拆成 6 个必须满足的点：

1. 数据集根目录变成 `data/`
2. split 使用 `data/splits/survival/.../{train.csv,test.csv}`
3. 训练不使用验证集（不构建 val_loader / 不做 per-epoch validation）
4. 评估只用最后一个 epoch：在 test set 计算 C-index
5. WSI 特征使用 TITAN（768 维）
6. RNA 使用 `data/data_csvs/rna/hallmarks/<CANCER>/rna_clean.csv`（按病人对齐）

这些点在代码里的“落点”分别是：

- **入口脚本**：决定训练/评估策略（是否有 val、何时算 C-index、读哪里的 split）
- **Dataset**：决定 WSI/RNA/生存标签从哪里读、如何对齐、`__getitem__` 返回什么
- **collate_fn**：决定 DataLoader 如何组 batch（Hallmarks=50 路 omic → 不能写死 6 路）
- **Trainer**：决定训练循环怎么 unpack batch、怎么把 50 路 omic 打包成 `x_omic1..x_omic50`
- **Model 输入维度**：TITAN=768，原版默认写死 1024，需要参数化

---

## 2. 原版 `MOTCat-main.zip`（官方）流程是什么？为什么它天然会用到验证集？

### 2.1 原版入口与数据假设

原版入口是 `main.py`（zip 内路径 `MOTCat-main/main.py`）。它的默认数据假设是：

- **WSI 特征**：来自 CLAM/类似流程提取的 patch-level `.pt`（通常每个 patch 1024 维），存在 `<DATA_ROOT_DIR>/pt_files/*.pt`
- **RNA/Genomics**：使用 repo 自带的 `dataset_csv/` 或 `datasets_csv_sig/`（MCAT 预处理结果）
- **split**：使用 repo 自带的 `splits/5foldcv/...`（train/val 交叉验证划分）

### 2.2 原版训练为何必然“有验证集”

原版的核心训练函数是 `utils/core_utils.py::train()`（zip 内 `MOTCat-main/utils/core_utils.py`）。它的结构（简化）是：

```python
train_loader = ...
val_loader = ...
monitor_cindex = Monitor_CIndex()  # 监控 val c-index

for epoch in range(max_epochs):
    train_loop(..., train_loader)
    val_latest, c_index_val, stop = validate_survival_*(..., val_loader, monitor_cindex)

    # 用 val c-index 选 best epoch，并保存 checkpoint
    if c_index_val > max_c_index:
        save_checkpoint(...)
```

因此原版的“模型选择/报告指标”逻辑是：

- **每个 epoch 都跑一次验证集**
- **用验证集 c-index 选最好的 epoch**（而不是用最后 epoch）

这与你的设定（无验证集 + 最后 epoch test）不同。

---

## 3. 当前仓库相对 `MOTCat-main.zip` 改了什么？（你现在有两条并存的流程）

对比 zip 与当前目录后，可以把仓库理解为两条流程并存：

### 3.1 原版流程：仍然保留（带 val）

这些文件与 zip 完全一致（也就是仍然是原版带验证集）：  

- `main.py`
- `utils/core_utils.py`
- `trainer/coattn_trainer.py`

含义：如果你继续运行 `main.py`，你仍然在跑“原版带验证集”的流程。

### 3.2 新增流程：你为了 `data/` + TITAN + Hallmarks 新增的第二条流程

新增（zip 中没有）：  

- `main_hallmarks.py`：新入口（读取 `data/`，不构建 val，只在最后 epoch 测 test）
- `trainer/hallmarks_trainer.py`：新 trainer（支持 50 路 Hallmark 通路输入）

修改（zip 中有，但内容变了）：  

- `dataset/dataset_survival.py`：新增 Hallmarks/TITAN Dataset 与 Split 类
- `utils/utils.py`：新增 `coattn_hallmarks` 对应的 collate_fn
- `models/model_motcat.py`：`wsi_dim` 参数化（固定 1024 → 可配置）
- `models/model_coattn.py`：`wsi_dim` 参数化（固定 1024 → 可配置）

---

## 4. 关键：到底是哪段代码实现了“无验证集 + 最后 epoch test C-index”？

答案：**`main_hallmarks.py` 的训练结构决定的**。

### 4.1 原版（zip）逻辑：每 epoch validate + 选 best val

入口：`main.py` → `utils/core_utils.py::train()`  
特点：构建 `train_loader` + `val_loader`；每 epoch 都 validate；用 val c-index 保存 best checkpoint。

### 4.2 新版（你现在）逻辑：只训练，不 validate；最后一次对 test 算 C-index

入口：`main_hallmarks.py`（zip 中没有）

它在 fold 内的核心逻辑是：

```python
train_loader = ...
test_loader = ...

# 训练：只跑 train loop
for epoch in range(max_epochs):
    train_loop_survival_hallmarks*(..., train_loader)

# 评估：训练结束后只跑一次 test（epoch=max_epochs-1）
test_results, c_index_test = validate_survival_hallmarks*(epoch=max_epochs-1, loader=test_loader)
```

因此：

- 没有 val_loader → **自然就是“无验证集”**
- C-index 只在训练结束后在 test 算一次 → **自然就是“最后 epoch test C-index”**

---

## 5. 新增的 `data/` 数据集是如何跑通的？（文件 → tensor → model 的数据流）

你现在的新数据目录（最小必需结构）是：

```
data/
  titan_features/
    TCGA_TITAN_features.pkl
  titan_embeddings/
    *.pt
  data_csvs/
    rna/
      metadata/
        hallmarks_signatures.csv
      hallmarks/
        <CANCER>/
          rna_clean.csv
  splits/
    survival/
      TCGA_<CANCER>_overall_survival_k=<fold>/
        train.csv
        test.csv
```

这条新流程的数据流是：

1. `main_hallmarks.py` 创建 base dataset（一次性加载 TITAN + RNA + signatures）
2. 每个 fold：读取 `train.csv/test.csv` → 生成 `train_split/test_split`
3. DataLoader 使用 `mode='coattn_hallmarks'` 的 collate，把 50 路 omic 组织成一个 batch
4. trainer 动态 unpack batch，构造 `x_omic1..x_omic50` 传给模型
5. 训练跑满 epoch 后，在 test 上计算 C-index

---

## 6. 逐文件解释：相对原版 zip，你是如何“在原基础上”改到满足需求的？

### 6.1 新增 `main_hallmarks.py`：新的训练入口（决定 split/无验证集/最后 epoch test）

你新增这个入口脚本的意义在于：**不破坏原版 `main.py`**，但能完全按新数据结构和新实验设定跑一条独立流程。

它做了这些关键事情：

1. **读取 split**
   - 每折从 `data/splits/survival/TCGA_<CANCER>_overall_survival_k=<fold>/{train.csv,test.csv}` 读取
2. **创建 split dataset**
   - `train_split = Generic_MIL_Survival_Split_Hallmarks(base_dataset, train_csv)`
   - `test_split  = Generic_MIL_Survival_Split_Hallmarks(base_dataset, test_csv)`
3. **不创建 val_loader**
   - 只创建 `train_loader` + `test_loader`
4. **训练循环只 train**
   - `for epoch in range(max_epochs): train_loop_survival_hallmarks*()`
5. **训练结束只测一次 test**
   - `validate_survival_hallmarks*(epoch=max_epochs-1, loader=test_loader)` → 得到 test c-index
6. **保存最终模型**
   - 保存 `s_<fold>_final.pt`（而不是“best val 的 checkpoint”）

这就是“无验证集 + 最后 epoch test”的核心实现。

### 6.2 修改 `dataset/dataset_survival.py`：新增 Hallmarks/TITAN 数据集类（实现新 `data/` 的读取与对齐）

原版 zip 的 `dataset/dataset_survival.py` 只有 CLAM/MCAT 的数据读法（从 `pt_files/*.pt` 读 WSI patch 特征，基因组来自 `dataset_csv`）。

你现在版本新增了两个类（概念上是“新数据集分支”，不影响原版类）：  

- `Generic_MIL_Survival_Dataset_Hallmarks`
- `Generic_MIL_Survival_Split_Hallmarks`

它们实现了这几件事：

**(1) TITAN WSI 特征读取（slide-level，768 维）**

- 从 `data/titan_features/TCGA_TITAN_features.pkl` 读取
- pkl 里有两个关键键：`filenames` 和 `embeddings`
  - `filenames`：slide_id（不带 `.svs`）
  - `embeddings`：与之对应的 768 维向量
- 构造 `slide_to_embedding[slide_id] = embedding_768`

**(2) Split CSV 读取与病人聚合**

- split CSV 是你现在这套 `train.csv/test.csv`（包含很多临床字段）
- 代码主要用到：`case_id` / `slide_id` / `os_survival_days` / `os_censorship`
- 通过 `case_id -> [slide_id...]` 建 `patient_dict`
- 取 `case_id` 去重得到患者级 `patients_df`

**(3) RNA Hallmarks 数据读取与对齐**

- RNA 文件：`data/data_csvs/rna/hallmarks/<CANCER>/rna_clean.csv`
- 对齐键：`case_id = sample[:12]`
- Hallmark 通路基因集：`data/data_csvs/rna/metadata/hallmarks_signatures.csv`（50 列=50 条通路）
- 最终每个病人产生 50 个 omic 输入（每个是“该通路基因的表达向量”）

**(4) `__getitem__` 返回的样本结构**

Hallmarks dataset 的 `__getitem__` 返回的是：

- `x_path`：shape `[num_slides, 768]`（把这个病人的多张 slide embedding stack 起来）
- `x_omic1..x_omic50`：每个是 1D tensor（每条通路一个向量）
- `label`：离散后的时间 bin（0..3）
- `event_time`：`os_survival_days`
- `c`：`os_censorship`

这就是后续 trainer/model 能跑起来的关键。

### 6.3 修改 `utils/utils.py`：新增 Hallmarks 的 collate + 新的 dataloader mode

原版 zip 的 coattn 流程默认是 **6 路 omic**，所以 collate/trainer 都写死了 `omic1..omic6`。

你现在新增：

- `collate_MIL_survival_sig_hallmarks`：可以处理“任意数量 omic”（这里是 50）
- `get_split_loader(..., mode='coattn_hallmarks')`：让 DataLoader 走新的 collate

这样 DataLoader 才能产出一个 batch：`[wsi] + [omic1..omicN] + [label,event_time,c]`。

### 6.4 新增 `trainer/hallmarks_trainer.py`：动态 unpack 50 路 omic 并计算 C-index

原版 `trainer/coattn_trainer.py` 写死 unpack：

```python
for batch_idx, (data_WSI, data_omic1, ..., data_omic6, label, event_time, c) in enumerate(loader):
    ...
```

Hallmarks 有 50 路，所以你新增了 trainer，它的核心思路是：

1. WSI 固定在 `batch_data[0]`
2. `label/event_time/c` 固定在最后 3 个
3. 中间 `batch_data[1:-3]` 全部当作 omic list
4. 循环构造：

```python
kwargs = {'x_path': data_WSI}
for i, omic in enumerate(omics):
    kwargs[f'x_omic{i+1}'] = omic
hazards, S, Y_hat, A = model(**kwargs)
```

并在评估时用：

- `risk = -sum(S)` 作为风险分数
- `concordance_index_censored(...)` 计算 C-index

在 `main_hallmarks.py` 里，它被“只在最后一次”用于 test 评估，从而实现你的指标需求。

### 6.5 修改 `models/model_motcat.py` / `models/model_coattn.py`：让 WSI 输入维度从 1024 变为可配置（支持 TITAN=768）

原版 zip 的两份模型都把 WSI 输入写死 1024。例如 zip 内 `models/model_motcat.py` 的关键片段是：

```python
def __init__(..., ot_impl=\"pot-uot-l2\"):
    self.size_dict_WSI = {\"small\": [1024, 256, 256], \"big\": [1024, 512, 384]}
```

你现在的版本把它参数化为（新版片段）：

```python
def __init__(..., ot_impl=\"pot-uot-l2\", wsi_dim=1024):
    self.size_dict_WSI = {\"small\": [wsi_dim, 256, 256], \"big\": [wsi_dim, 512, 384]}
```

然后在 `main_hallmarks.py` 里创建模型时传入：

- `wsi_dim=768`

从而避免 TITAN 768 维输入在第一层 `Linear(1024,256)` 处维度不匹配。

---

## 7. 你问的关键点：TITAN 用 `titan_embeddings` 还是 `titan_features`？

### 7.1 当前代码用的是哪一个？

当前 Hallmarks/TITAN 流程用的是：

- `data/titan_features/TCGA_TITAN_features.pkl` 里的 `embeddings`

原因：你的 split CSV 用的是 `slide_id`（带 `-01Z-00-DX...UUID`），而这个 pkl 的 `filenames` 正好是 slide_id（匹配成本最低）。

### 7.2 那 `titan_embeddings/*.pt` 是什么？

你目录下 `data/titan_embeddings/*.pt` 里的每个文件基本是一个 768 维向量，但文件名更像 `case_id`（例如 `TCGA-02-0001.pt`）。

因此在你当前 “split 以 `slide_id` 对齐 WSI” 的前提下：

- `titan_embeddings` **不能直接按 `slide_id` 索引**
- 若坚持用它，需要改动对齐策略（从 slide_id 改为 case_id，并定义多 slide 聚合），这不是当前实现

结论：按你现在的数据结构与实现，选 `titan_features/TCGA_TITAN_features.pkl` 是正确的。

---

## 8. 如何运行你这条新流程？

你要跑的是新入口：

- `main_hallmarks.py`

示例（PowerShell）：

```powershell
python main_hallmarks.py --data_dir .\\data --cancer_type BLCA --model_type motcat --max_epochs 20 --k 5 --seed 1
```

你会看到：

- 训练过程打印 train loss / train c-index（用于观察训练状态）
- 每个 fold 训练结束打印一次 `test c-index`
- 每个 fold 保存一次最终模型 `s_<fold>_final.pt`

---

## 9.（建议）你可以自己做的 sanity check

如果你想确认“对齐没有问题”，建议检查两件事：

1. split 里的 `slide_id` 是否都能在 `TCGA_TITAN_features.pkl` 的 `filenames` 里找到（否则会 fallback 0 向量）
2. split 里的 `case_id` 是否都能在 `rna_clean.csv` 的 `sample[:12]` 里找到（否则会 fallback 0 向量）

---

## 10.（可选）如何自己对比 zip（逐文件 diff）

你可以把 `MOTCat-main.zip` 解压到临时目录，然后用 diff 工具对比：

```powershell
Expand-Archive -Path .\\MOTCat-main.zip -DestinationPath .\\_orig -Force
```

然后对比重点文件即可，例如：

- `_orig\\MOTCat-main\\dataset\\dataset_survival.py` vs `dataset\\dataset_survival.py`
- `_orig\\MOTCat-main\\utils\\utils.py` vs `utils\\utils.py`
- `_orig\\MOTCat-main\\models\\model_motcat.py` vs `models\\model_motcat.py`
- `_orig\\MOTCat-main\\models\\model_coattn.py` vs `models\\model_coattn.py`
- zip 中没有 `main_hallmarks.py`、`trainer\\hallmarks_trainer.py`

