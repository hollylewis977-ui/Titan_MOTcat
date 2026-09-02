from __future__ import print_function, division
import math
import os
import pdb
import pickle
import re

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

from utils.utils import generate_split, nth


###############################################################################
# New Hallmarks Dataset Class for TITAN features + 50 Hallmark pathways
###############################################################################
class Generic_MIL_Survival_Dataset_Hallmarks(Dataset):
    """
    Dataset class for survival prediction using:
    - TITAN slide-level features (768-dim per slide)
    - 50 Hallmark pathway gene signatures
    - New data format with os_survival_days and dss_censorship
    """
    def __init__(self,
                 data_dir,                    # Base data directory
                 cancer_type,                 # e.g., 'BLCA', 'BRCA'
                 mode='coattn',
                 n_bins=4,
                 eps=1e-6,
                 print_info=True):

        self.data_dir = data_dir
        self.cancer_type = cancer_type
        self.mode = mode
        self.n_bins = n_bins
        self.eps = eps
        self.label_col = 'dss_survival_days'

        # Load TITAN features
        titan_pkl_path = os.path.join(data_dir, 'titan_features', 'TCGA_TITAN_features.pkl')
        print(f"Loading TITAN features from {titan_pkl_path}...")
        with open(titan_pkl_path, 'rb') as f:
            titan_data = pickle.load(f)

        # Create slide_id -> embedding mapping
        # filename format: TCGA-06-1087-01Z-00-DX2.1f91f05a-f277-4c98-9955-37e0c83b745f
        self.slide_to_embedding = {}
        for i, fname in enumerate(titan_data['filenames']):
            self.slide_to_embedding[fname] = titan_data['embeddings'][i]
        print(f"Loaded {len(self.slide_to_embedding)} slide embeddings")

        # Load RNA data
        rna_csv_path = os.path.join(data_dir, 'data_csvs', 'rna', 'hallmarks', cancer_type, 'rna_clean.csv')
        print(f"Loading RNA data from {rna_csv_path}...")
        self.rna_df = pd.read_csv(rna_csv_path)

        # Create case_id from sample (first 12 chars)
        self.rna_df['case_id'] = self.rna_df['sample'].str[:12]

        # Extract sample type from TCGA barcode (chars 13-15, e.g., "-01" for primary tumor)
        # TCGA sample types: -01=Primary Tumor, -02=Recurrent, -06=Metastatic, -11=Normal
        # We only want primary tumor samples (-01) for survival prediction
        self.rna_df['sample_type'] = self.rna_df['sample'].str[12:15]
        n_before = len(self.rna_df)
        self.rna_df = self.rna_df[self.rna_df['sample_type'] == '-01'].copy()
        n_after = len(self.rna_df)
        print(f"Filtered to primary tumor (-01) samples: {n_before} -> {n_after}")

        # Verify no duplicate case_ids after filtering
        n_unique_cases = self.rna_df['case_id'].nunique()
        if n_after != n_unique_cases:
            print(f"Warning: {n_after - n_unique_cases} duplicate case_ids remain after filtering")
            # Keep first occurrence if duplicates exist
            self.rna_df = self.rna_df.drop_duplicates(subset='case_id', keep='first')
            print(f"After deduplication: {len(self.rna_df)} samples")

        # Set case_id as index for fast lookup
        self.rna_df_indexed = self.rna_df.set_index('case_id')
        print(f"Loaded RNA data for {len(self.rna_df)} unique patients")

        # Load Hallmarks signatures (50 pathways)
        signatures_csv_path = os.path.join(data_dir, 'data_csvs', 'rna', 'metadata', 'hallmarks_signatures.csv')
        print(f"Loading Hallmarks signatures from {signatures_csv_path}...")
        self.signatures = pd.read_csv(signatures_csv_path)

        # Get gene columns from RNA data (excluding metadata columns)
        gene_columns = [c for c in self.rna_df.columns if c not in ['Unnamed: 0', 'sample', 'case_id']]

        # Build omic names for each hallmark pathway
        self.omic_names = []
        for col in self.signatures.columns:
            genes = self.signatures[col].dropna().tolist()
            # Filter to genes that exist in RNA data
            valid_genes = [g for g in genes if g in gene_columns]
            self.omic_names.append(valid_genes)

        self.omic_sizes = [len(genes) for genes in self.omic_names]
        print(f"Loaded {len(self.omic_sizes)} Hallmark pathways with sizes: {self.omic_sizes[:5]}...")

        # These will be set when loading splits
        self.slide_data = None
        self.patient_dict = None
        self.num_classes = 2 * n_bins  # (bin, censorship) pairs

    def load_split(self, split_csv_path, external_bins=None):
        """Load a train or test split CSV

        Args:
            split_csv_path: Path to the split CSV file
            external_bins: Pre-computed bins from train split.
                          If None, compute bins from this split (for train).
                          If provided, use these bins (for test).
        """
        print(f"Loading split from {split_csv_path}...")
        self.slide_data = pd.read_csv(split_csv_path)

        # Build patient_dict: case_id -> list of slide_ids
        self.patient_dict = {}
        for idx, row in self.slide_data.iterrows():
            case_id = row['case_id']
            slide_id = row['slide_id']
            # Extract the part before .svs for matching with TITAN
            slide_id_clean = slide_id.rstrip('.svs') if slide_id.endswith('.svs') else slide_id

            if case_id not in self.patient_dict:
                self.patient_dict[case_id] = []
            self.patient_dict[case_id].append(slide_id_clean)

        # Get unique patients (case_ids)
        self.patients_df = self.slide_data.drop_duplicates(['case_id']).reset_index(drop=True)

        # Determine bins: use external_bins if provided, otherwise compute from this split
        if external_bins is not None:
            # Use pre-computed bins from train split (for test/val)
            q_bins = external_bins
            print(f"Using external bins: {q_bins}")
        else:
            # Compute bins from this split's uncensored patients (for train)
            uncensored_df = self.patients_df[self.patients_df['dss_censorship'] < 1]

            if len(uncensored_df) > 0:
                _, q_bins = pd.qcut(uncensored_df[self.label_col], q=self.n_bins, retbins=True, labels=False)
                # Extend bins to cover all possible values
                q_bins[-1] = self.patients_df[self.label_col].max() + self.eps
                q_bins[0] = self.patients_df[self.label_col].min() - self.eps
            else:
                # Fallback if no uncensored samples
                q_bins = np.linspace(
                    self.patients_df[self.label_col].min() - self.eps,
                    self.patients_df[self.label_col].max() + self.eps,
                    self.n_bins + 1
                )
            print(f"Computed bins from train: {q_bins}")

        # Apply bins to all patients
        # Handle edge cases where test values may fall outside train bins
        disc_labels = pd.cut(
            self.patients_df[self.label_col],
            bins=q_bins,
            labels=False,
            right=False,
            include_lowest=True
        )

        # Handle NaN labels (values outside bin range) by clipping
        disc_labels = disc_labels.fillna(self.n_bins - 1)  # Assign to last bin if out of range

        self.patients_df = self.patients_df.copy()
        self.patients_df['disc_label'] = disc_labels.values.astype(int)

        # Clip to valid range [0, n_bins-1]
        self.patients_df['disc_label'] = self.patients_df['disc_label'].clip(0, self.n_bins - 1)

        # Build label dictionary
        self.label_dict = {}
        key_count = 0
        for i in range(self.n_bins):
            for c in [0, 1]:
                self.label_dict[(i, c)] = key_count
                key_count += 1

        # Create combined label
        self.patients_df['label'] = self.patients_df.apply(
            lambda row: self.label_dict[(row['disc_label'], int(row['dss_censorship']))], axis=1
        )

        self.bins = q_bins
        print(f"Loaded {len(self.patients_df)} patients")

        return self

    def __len__(self):
        return len(self.patients_df)

    def get_scaler(self):
        """Get StandardScaler fitted on genomic features"""
        # Collect all genomic features for this split
        all_genes = []
        for genes in self.omic_names:
            all_genes.extend(genes)
        all_genes = list(set(all_genes))

        # Get genomic data for all patients in this split
        genomic_data = []
        for idx in range(len(self.patients_df)):
            case_id = self.patients_df.iloc[idx]['case_id']
            if case_id in self.rna_df_indexed.index:
                rna_row = self.rna_df_indexed.loc[case_id]
                if isinstance(rna_row, pd.DataFrame):
                    rna_row = rna_row.iloc[0]
                genomic_data.append(rna_row[all_genes].values)

        if len(genomic_data) > 0:
            genomic_data = np.array(genomic_data)
            scaler = StandardScaler().fit(genomic_data)
        else:
            scaler = None

        return (scaler, all_genes)

    def apply_scaler(self, scalers):
        """Apply pre-fitted scaler to genomic features (one-time transform like original MOTCat)"""
        scaler = scalers[0]
        scaler_genes = scalers[1]
        scaler_gene_set = set(scaler_genes)  # O(1) lookup

        # Build genomic matrix for all patients in this split
        genomic_data = []
        for idx in range(len(self.patients_df)):
            case_id = self.patients_df.iloc[idx]['case_id']
            if case_id in self.rna_df_indexed.index:
                rna_row = self.rna_df_indexed.loc[case_id]
                if isinstance(rna_row, pd.DataFrame):
                    rna_row = rna_row.iloc[0]
                genomic_data.append(rna_row[scaler_genes].values)
            else:
                genomic_data.append(np.zeros(len(scaler_genes)))

        # One-shot transform + convert to float32
        genomic_data = np.array(genomic_data, dtype=np.float64)
        if scaler is not None:
            genomic_data = scaler.transform(genomic_data)
        self.genomic_matrix = genomic_data.astype(np.float32)  # (n_patients, n_genes)

        # Pre-compute column indices for each hallmark (avoid string lookups in __getitem__)
        gene_to_col = {g: i for i, g in enumerate(scaler_genes)}
        self.hallmark_col_indices = []
        for genes in self.omic_names:
            valid_indices = [gene_to_col[g] for g in genes if g in scaler_gene_set]
            if len(valid_indices) > 0:
                self.hallmark_col_indices.append(np.array(valid_indices, dtype=np.int32))
            else:
                self.hallmark_col_indices.append(None)

    def __getitem__(self, idx):
        patient_row = self.patients_df.iloc[idx]
        case_id = patient_row['case_id']
        label = patient_row['disc_label']
        event_time = patient_row[self.label_col]
        c = patient_row['dss_censorship']

        # Get all slide_ids for this patient
        slide_ids = self.patient_dict.get(case_id, [])

        # Load WSI features (concatenate multiple slides)
        wsi_features = []
        for slide_id in slide_ids:
            if slide_id in self.slide_to_embedding:
                emb = self.slide_to_embedding[slide_id]
                wsi_features.append(torch.from_numpy(emb).float())

        if len(wsi_features) > 0:
            wsi_features = torch.stack(wsi_features)  # [num_slides, 768]
        else:
            wsi_features = torch.zeros(1, 768)

        # Load genomic features - just numpy slicing on pre-scaled matrix (fast!)
        omics = []
        for col_indices in self.hallmark_col_indices:
            if col_indices is not None:
                omic_values = self.genomic_matrix[idx, col_indices]  # numpy advanced indexing
                omics.append(torch.from_numpy(omic_values))
            else:
                omics.append(torch.zeros(1))

        if self.mode == 'coattn':
            return (wsi_features,) + tuple(omics) + (label, event_time, c, case_id)
        else:
            raise NotImplementedError(f"Mode {self.mode} not implemented for Hallmarks dataset")


class Generic_MIL_Survival_Split_Hallmarks(Generic_MIL_Survival_Dataset_Hallmarks):
    """A split (train/test) view of the Hallmarks dataset"""
    def __init__(self, parent_dataset, split_csv_path, external_bins=None):
        """
        Args:
            parent_dataset: The parent Generic_MIL_Survival_Dataset_Hallmarks
            split_csv_path: Path to train.csv or test.csv
            external_bins: Pre-computed bins from train split.
                          - For train split: pass None (will compute bins)
                          - For test split: pass train_split.bins (will use train bins)
        """
        # Copy attributes from parent
        self.data_dir = parent_dataset.data_dir
        self.cancer_type = parent_dataset.cancer_type
        self.mode = parent_dataset.mode
        self.n_bins = parent_dataset.n_bins
        self.eps = parent_dataset.eps
        self.label_col = parent_dataset.label_col
        self.slide_to_embedding = parent_dataset.slide_to_embedding
        self.rna_df = parent_dataset.rna_df
        self.rna_df_indexed = parent_dataset.rna_df_indexed
        self.signatures = parent_dataset.signatures
        self.omic_names = parent_dataset.omic_names
        self.omic_sizes = parent_dataset.omic_sizes
        self.num_classes = parent_dataset.num_classes

        # Load this specific split with optional external bins
        self.load_split(split_csv_path, external_bins=external_bins)

        # Build slide_cls_ids for weighted sampling
        self._build_slide_cls_ids()

    def _build_slide_cls_ids(self):
        """Build slide_cls_ids for weighted sampling based on survival labels"""
        self.slide_cls_ids = {}
        for cls in range(self.num_classes):
            self.slide_cls_ids[cls] = []

        for idx in range(len(self.patients_df)):
            label = self.patients_df.iloc[idx]['label']
            self.slide_cls_ids[label].append(idx)

    def getlabel(self, idx):
        return self.patients_df.iloc[idx]['label']


class Generic_WSI_Survival_Dataset(Dataset):
    def __init__(self, csv_path = 'dataset_csv/ccrcc_clean.csv', mode = 'omic', apply_sig = False, 
        shuffle = False, seed = 7, print_info = True, n_bins = 4, ignore=[],
        patient_strat=False, label_col = None, filter_dict = {}, eps=1e-6):
        r"""
        Generic_WSI_Survival_Dataset 

        Args:
            csv_file (string): Path to the csv file with annotations.
            shuffle (boolean): Whether to shuffle
            seed (int): random seed for shuffling the data
            print_info (boolean): Whether to print a summary of the dataset
            label_dict (dict): Dictionary with key, value pairs for converting str labels to int
            ignore (list): List containing class labels to ignore
        """
        self.custom_test_ids = None
        self.seed = seed
        self.print_info = print_info
        self.patient_strat = patient_strat
        self.train_ids, self.val_ids, self.test_ids  = (None, None, None)
        self.data_dir = None

        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(slide_data)


        slide_data = pd.read_csv(csv_path, low_memory=False)
        ### new
        missing_slides_ls = ['TCGA-A7-A6VX-01Z-00-DX2.9EE94B59-6A2C-4507-AA4F-DC6402F2B74F.svs',
                             'TCGA-A7-A0CD-01Z-00-DX2.609CED8D-5947-4753-A75B-73A8343B47EC.svs',
                             'TCGA-HT-7483-01Z-00-DX1.7241DF0C-1881-4366-8DD9-11BF8BDD6FBF.svs',
                             'TCGA-06-0882-01Z-00-DX2.7ad706e3-002e-4e29-88a9-18953ba422bf.svs']
        slide_data.drop(slide_data[slide_data['slide_id'].isin(missing_slides_ls)].index, inplace=True)

        #slide_data = slide_data.drop(['Unnamed: 0'], axis=1)
        if 'case_id' not in slide_data:
            slide_data.index = slide_data.index.str[:12]
            slide_data['case_id'] = slide_data.index
            slide_data = slide_data.reset_index(drop=True)
        

        if not label_col:
            label_col = 'survival_months'
        else:
            assert label_col in slide_data.columns
        self.label_col = label_col

        if "IDC" in slide_data['oncotree_code']: # must be BRCA (and if so, use only IDCs)
            slide_data = slide_data[slide_data['oncotree_code'] == 'IDC']

        patients_df = slide_data.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        disc_labels, q_bins = pd.qcut(uncensored_df[label_col], q=n_bins, retbins=True, labels=False)
        q_bins[-1] = slide_data[label_col].max() + eps
        q_bins[0] = slide_data[label_col].min() - eps
        
        disc_labels, q_bins = pd.cut(patients_df[label_col], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'label', disc_labels.values.astype(int))

        patient_dict = {}
        slide_data = slide_data.set_index('case_id')
        for patient in patients_df['case_id']:
            slide_ids = slide_data.loc[patient, 'slide_id']
            if isinstance(slide_ids, str):
                slide_ids = np.array(slide_ids).reshape(-1)
            else:
                slide_ids = slide_ids.values
            patient_dict.update({patient:slide_ids})

        self.patient_dict = patient_dict
    
        slide_data = patients_df
        slide_data.reset_index(drop=True, inplace=True)
        slide_data = slide_data.assign(slide_id=slide_data['case_id'])

        label_dict = {}
        key_count = 0
        for i in range(len(q_bins)-1):
            for c in [0, 1]:
                print('{} : {}'.format((i, c), key_count))
                label_dict.update({(i, c):key_count})
                key_count+=1

        self.label_dict = label_dict
        for i in slide_data.index:
            key = slide_data.loc[i, 'label']
            slide_data.at[i, 'disc_label'] = key
            censorship = slide_data.loc[i, 'censorship']
            key = (key, int(censorship))
            slide_data.at[i, 'label'] = label_dict[key]

        self.bins = q_bins
        self.num_classes=len(self.label_dict)
        patients_df = slide_data.drop_duplicates(['case_id'])
        self.patient_data = {'case_id':patients_df['case_id'].values, 'label':patients_df['label'].values}

        new_cols = list(slide_data.columns[-2:]) + list(slide_data.columns[:-2])
        slide_data = slide_data[new_cols]
        self.slide_data = slide_data
        self.metadata = slide_data.columns[:12]
        self.mode = mode
        self.cls_ids_prep()


        ### Signatures
        self.apply_sig = apply_sig
        if self.apply_sig:
            self.signatures = pd.read_csv('./datasets_csv_sig/signatures.csv')
        else:
            self.signatures = None

        if print_info:
            self.summarize()


    def cls_ids_prep(self):
        self.patient_cls_ids = [[] for i in range(self.num_classes)]        
        for i in range(self.num_classes):
            self.patient_cls_ids[i] = np.where(self.patient_data['label'] == i)[0]

        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]


    def patient_data_prep(self):
        patients = np.unique(np.array(self.slide_data['case_id'])) # get unique patients
        patient_labels = []
        
        for p in patients:
            locations = self.slide_data[self.slide_data['case_id'] == p].index.tolist()
            assert len(locations) > 0
            label = self.slide_data['label'][locations[0]] # get patient label
            patient_labels.append(label)
        
        self.patient_data = {'case_id':patients, 'label':np.array(patient_labels)}


    @staticmethod
    def df_prep(data, n_bins, ignore, label_col):
        mask = data[label_col].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        disc_labels, bins = pd.cut(data[label_col], bins=n_bins)
        return data, bins

    def __len__(self):
        if self.patient_strat:
            return len(self.patient_data['case_id'])
        else:
            return len(self.slide_data)

    def summarize(self):
        print("label column: {}".format(self.label_col))
        print("label dictionary: {}".format(self.label_dict))
        print("number of classes: {}".format(self.num_classes))
        print("slide-level counts: ", '\n', self.slide_data['label'].value_counts(sort = False))
        for i in range(self.num_classes):
            print('Patient-LVL; Number of samples registered in class %d: %d' % (i, self.patient_cls_ids[i].shape[0]))
            print('Slide-LVL; Number of samples registered in class %d: %d' % (i, self.slide_cls_ids[i].shape[0]))


    def get_split_from_df(self, all_splits: dict, split_key: str='train', scaler=None):
        split = all_splits[split_key]
        split = split.dropna().reset_index(drop=True)

        if len(split) > 0:
            mask = self.slide_data['slide_id'].isin(split.tolist())
            df_slice = self.slide_data[mask].reset_index(drop=True)
            split = Generic_Split(df_slice, metadata=self.metadata, mode=self.mode, 
                                  signatures=self.signatures, data_dir=self.data_dir, 
                                  label_col=self.label_col, patient_dict=self.patient_dict, num_classes=self.num_classes)
        else:
            split = None
        
        return split


    def return_splits(self, from_id: bool=True, csv_path: str=None):
        if from_id:
            raise NotImplementedError
        else:
            assert csv_path 
            all_splits = pd.read_csv(csv_path)
            train_split = self.get_split_from_df(all_splits=all_splits, split_key='train')
            val_split = self.get_split_from_df(all_splits=all_splits, split_key='val')
            test_split = None #self.get_split_from_df(all_splits=all_splits, split_key='test')

            ### --> Normalizing Data
            print("****** Normalizing Data ******")
            scalers = train_split.get_scaler()
            train_split.apply_scaler(scalers=scalers)
            val_split.apply_scaler(scalers=scalers)
            #test_split.apply_scaler(scalers=scalers)
            ### <--
        return train_split, val_split#, test_split


    def get_list(self, ids):
        return self.slide_data['slide_id'][ids]

    def getlabel(self, ids):
        return self.slide_data['label'][ids]

    def __getitem__(self, idx):
        return None

    def __getitem__(self, idx):
        return None


class Generic_MIL_Survival_Dataset(Generic_WSI_Survival_Dataset):
    def __init__(self, data_dir, mode: str='omic', **kwargs):
        super(Generic_MIL_Survival_Dataset, self).__init__(**kwargs)
        self.data_dir = data_dir
        self.mode = mode
        self.use_h5 = False

    def load_from_h5(self, toggle):
        self.use_h5 = toggle

    def __getitem__(self, idx):
        case_id = self.slide_data['case_id'][idx]
        label = self.slide_data['disc_label'][idx]
        event_time = self.slide_data[self.label_col][idx]
        c = self.slide_data['censorship'][idx]
        slide_ids = self.patient_dict[case_id]

        if type(self.data_dir) == dict:
            source = self.slide_data['oncotree_code'][idx]
            data_dir = self.data_dir[source]
        else:
            data_dir = self.data_dir
        
        if not self.use_h5:
            if self.data_dir:
                if self.mode == 'path':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path,map_location=torch.device('cpu'))
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    return (path_features, torch.zeros((1,1)), label, event_time, c)

                elif self.mode == 'cluster':
                    path_features = []
                    cluster_ids = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path,map_location=torch.device('cpu'))
                        path_features.append(wsi_bag)
                        cluster_ids.extend(self.fname2ids[slide_id[:-4]+'.pt'])
                    path_features = torch.cat(path_features, dim=0)
                    cluster_ids = torch.Tensor(cluster_ids)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, cluster_ids, genomic_features, label, event_time, c)

                elif self.mode == 'omic':
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (torch.zeros((1,1)), genomic_features, label, event_time, c)

                elif self.mode == 'pathomic':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path,map_location=torch.device('cpu'))
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    genomic_features = torch.tensor(self.genomic_features.iloc[idx])
                    return (path_features, genomic_features, label, event_time, c)

                elif self.mode == 'coattn':
                    path_features = []
                    for slide_id in slide_ids:
                        wsi_path = os.path.join(data_dir, 'pt_files', '{}.pt'.format(slide_id.rstrip('.svs')))
                        wsi_bag = torch.load(wsi_path,map_location=torch.device('cpu'))
                        path_features.append(wsi_bag)
                    path_features = torch.cat(path_features, dim=0)
                    omic1 = torch.tensor(self.genomic_features[self.omic_names[0]].iloc[idx].values)
                    omic2 = torch.tensor(self.genomic_features[self.omic_names[1]].iloc[idx].values)
                    omic3 = torch.tensor(self.genomic_features[self.omic_names[2]].iloc[idx].values)
                    omic4 = torch.tensor(self.genomic_features[self.omic_names[3]].iloc[idx].values)
                    omic5 = torch.tensor(self.genomic_features[self.omic_names[4]].iloc[idx].values)
                    omic6 = torch.tensor(self.genomic_features[self.omic_names[5]].iloc[idx].values)
                    return (path_features, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c)
                else:
                    raise NotImplementedError('Mode [%s] not implemented.' % self.mode)
                ### <--
            else:
                return slide_ids, label, event_time, c


class Generic_Split(Generic_MIL_Survival_Dataset):
    def __init__(self, slide_data, metadata, mode, signatures=None, data_dir=None, label_col=None, patient_dict=None, num_classes=2):
        self.use_h5 = False
        self.slide_data = slide_data
        self.metadata = metadata
        self.mode = mode
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.label_col = label_col
        self.patient_dict = patient_dict
        self.slide_cls_ids = [[] for i in range(self.num_classes)]
        for i in range(self.num_classes):
            self.slide_cls_ids[i] = np.where(self.slide_data['label'] == i)[0]

        ### --> Initializing genomic features in Generic Split
        self.genomic_features = self.slide_data.drop(self.metadata, axis=1)
        self.signatures = signatures


        def series_intersection(s1, s2):
            return pd.Series(list(set(s1) & set(s2)))
        
        
        if self.signatures is not None:
            self.omic_names = []
            for col in self.signatures.columns:
                omic = self.signatures[col].dropna().unique()
                omic = np.concatenate([omic+mode for mode in ['_mut', '_cnv', '_rnaseq']])
                omic = sorted(series_intersection(omic, self.genomic_features.columns))
                self.omic_names.append(omic)
            self.omic_sizes = [len(omic) for omic in self.omic_names]
        # print("Shape", self.genomic_features.shape)
        ### <--

    def __len__(self):
        return len(self.slide_data)

    ### --> Getting StandardScaler of self.genomic_features
    def get_scaler(self):
        scaler_omic = StandardScaler().fit(self.genomic_features)
        return (scaler_omic,)
    ### <--

    ### --> Applying StandardScaler to self.genomic_features
    def apply_scaler(self, scalers: tuple=None):
        transformed = pd.DataFrame(scalers[0].transform(self.genomic_features))
        transformed.columns = self.genomic_features.columns
        self.genomic_features = transformed
    ### <--