import torch
from torch import linalg as LA
import torch.nn.functional as F
import torch.nn as nn

import ot

from models.model_utils import SNN_Block, Attn_Net_Gated, BilinearFusion, FiLMGenerator


class OT_Attn_assem(nn.Module):
    def __init__(self,impl='pot-uot-l2',ot_reg=0.1, ot_tau=0.5) -> None:
        super().__init__()
        self.impl = impl
        self.ot_reg = ot_reg
        self.ot_tau = ot_tau
        print("ot impl: ", impl)
    
    def normalize_feature(self,x):
        x = x - x.min(-1)[0].unsqueeze(-1)
        return x

    def OT(self, weight1, weight2):
        """
        Parmas:
            weight1 : (N, D)
            weight2 : (M, D)
        
        Return:
            flow : (N, M)
            dist : (1, )
        """

        if self.impl == "pot-sinkhorn-l2":
            self.cost_map = torch.cdist(weight1, weight2)**2 # (N, M)
            
            src_weight = weight1.sum(dim=1) / weight1.sum()
            dst_weight = weight2.sum(dim=1) / weight2.sum()
            
            cost_map_detach = self.cost_map.detach()
            flow = ot.sinkhorn(a=src_weight.detach(), b=dst_weight.detach(), 
                                M=cost_map_detach/cost_map_detach.max(), reg=self.ot_reg)
            dist = self.cost_map * flow 
            dist = torch.sum(dist)
            return flow, dist
        
        elif self.impl == "pot-uot-l2":
            a, b = torch.from_numpy(ot.unif(weight1.size()[0]).astype('float64')).to(weight1.device), torch.from_numpy(ot.unif(weight2.size()[0]).astype('float64')).to(weight2.device)
            self.cost_map = torch.cdist(weight1, weight2)**2 # (N, M)
            
            cost_map_detach = self.cost_map.detach()
            M_cost = cost_map_detach/cost_map_detach.max()
            
            flow = ot.unbalanced.sinkhorn_knopp_unbalanced(a=a, b=b, 
                                M=M_cost.double(), reg=self.ot_reg,reg_m=self.ot_tau)
            flow = flow.type(torch.FloatTensor).cuda()
            
            dist = self.cost_map * flow # (N, M)
            dist = torch.sum(dist) # (1,) float
            return flow, dist
        
        else:
            raise NotImplementedError

        

    def forward(self,x,y):
        '''
        x: (N, 1, D)
        y: (M, 1, D)
        '''
        x = x.squeeze(1)  # (N, D) - only squeeze dim 1
        y = y.squeeze(1)  # (M, D) - only squeeze dim 1

        # Ensure 2D tensors (at least 1 sample)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if y.dim() == 1:
            y = y.unsqueeze(0)

        x = self.normalize_feature(x)
        y = self.normalize_feature(y)

        pi, dist = self.OT(x, y)
        return pi.T.unsqueeze(0).unsqueeze(0), dist

       
#############################
### MOTCAT Implementation ###
#############################
class MOTCAT_Surv(nn.Module):
    def __init__(self, fusion='concat', omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25,
                 ot_reg=0.1, ot_tau=0.5, ot_impl="pot-uot-l2", wsi_dim=768):
        super(MOTCAT_Surv, self).__init__()
        self.fusion = fusion
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.wsi_dim = wsi_dim
        self.hidden_dim = 256

        # ========== Dual projection layers ==========
        # proj_ot: for OT computation + FiLM modulation (maintains gradient flow)
        self.proj_ot = nn.Sequential(
            nn.Linear(wsi_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # proj_768_to_256: for 768-dim aggregation path
        self.proj_768_to_256 = nn.Sequential(
            nn.Linear(wsi_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # ========== Genomic SNN (unchanged) ==========
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

        # ========== OT Co-attention ==========
        self.coattn = OT_Attn_assem(impl=ot_impl, ot_reg=ot_reg, ot_tau=ot_tau)

        # ========== FiLM Generator (NEW) ==========
        self.film_generator = FiLMGenerator(
            context_dim=self.hidden_dim,
            feature_dim=self.hidden_dim,
            hidden_dim=128,
            dropout=dropout
        )

        # ========== Path Transformer (batch_first=True) ==========
        path_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=8, dim_feedforward=512,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
        self.path_attention_head = Attn_Net_Gated(L=self.hidden_dim, D=self.hidden_dim, dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        # ========== Omic Transformer (batch_first=True) ==========
        omic_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim, nhead=8, dim_feedforward=512,
            dropout=dropout, activation='relu', batch_first=True
        )
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)
        self.omic_attention_head = Attn_Net_Gated(L=self.hidden_dim, D=self.hidden_dim, dropout=dropout, n_classes=1)
        self.omic_rho = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        # ========== Fusion + Classifier ==========
        if self.fusion == 'concat':
            self.mm = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=self.hidden_dim, dim2=self.hidden_dim, scale_dim1=8, scale_dim2=8, mmhid=self.hidden_dim)
        else:
            self.mm = None

        self.classifier = nn.Linear(self.hidden_dim, n_classes)

    def forward(self, **kwargs):
        x_path = kwargs['x_path']  # (N_slide, 768)
        x_omic = [kwargs['x_omic%d' % i] for i in range(1, len(self.omic_sizes)+1)]

        N_slide = x_path.shape[0]
        X_768 = x_path  # Keep original 768-dim features

        # Step 1: Project to 256-dim
        X_256 = self.proj_ot(x_path)  # (N_slide, 256)

        # Step 2: Genomic SNN processing
        h_omic_list = [self.sig_networks[idx](sig_feat) for idx, sig_feat in enumerate(x_omic)]
        G_256 = torch.stack(h_omic_list)  # (M_omic, 256)

        # Step 3: OT computation for transport matrix P
        X_256_ot = X_256.unsqueeze(1)  # (N_slide, 1, 256)
        G_256_ot = G_256.unsqueeze(1)  # (M_omic, 1, 256)
        A_coattn, ot_dist = self.coattn(X_256_ot, G_256_ot)
        # A_coattn: (1, 1, M_omic, N_slide)

        P = A_coattn.squeeze(0).squeeze(0).T  # (N_slide, M_omic)
        if P.dim() == 1:
            P = P.unsqueeze(0)

        # Step 4: Compute RNA context
        # Use OT-style normalization (preserves transport matrix semantics)
        eps = 1e-8
        P_normalized = P / (P.sum(dim=1, keepdim=True) + eps)  # (N_slide, M_omic)
        C = torch.mm(P_normalized, G_256)  # (N_slide, 256)

        # Step 5: FiLM modulation (KEY gradient path!)
        gamma, beta = self.film_generator(C)
        X_256_mod = gamma * X_256 + beta  # (N_slide, 256)

        # Step 6: Path Transformer processes N_slide tokens
        X_256_mod_batch = X_256_mod.unsqueeze(0)  # (1, N_slide, 256)
        h_path_trans = self.path_transformer(X_256_mod_batch).squeeze(0)  # (N_slide, 256)

        # Step 7: Attention Pooling
        A_path, Z_256 = self.path_attention_head(h_path_trans)
        A_path = A_path.T  # (1, N_slide)
        w = F.softmax(A_path, dim=1)  # (1, N_slide)

        # Step 8: Dual-path aggregation
        h_path_768 = torch.mm(w, X_768)  # (1, 768)
        h_path_768_proj = self.proj_768_to_256(h_path_768)  # (1, 256)
        h_path_256 = torch.mm(w, Z_256)  # (1, 256)
        h_path = h_path_768_proj + h_path_256  # (1, 256)
        h_path = self.path_rho(h_path).squeeze()  # (256,)

        # Step 9: Omic Transformer
        G_256_batch = G_256.unsqueeze(0)  # (1, M_omic, 256)
        h_omic_trans = self.omic_transformer(G_256_batch).squeeze(0)  # (M_omic, 256)
        A_omic, h_omic_feat = self.omic_attention_head(h_omic_trans)
        A_omic = A_omic.T
        h_omic = torch.mm(F.softmax(A_omic, dim=1), h_omic_feat)  # (1, 256)
        h_omic = self.omic_rho(h_omic).squeeze()  # (256,)

        # Step 10: Fusion + Classification
        if self.fusion == 'bilinear':
            h = self.mm(h_path.unsqueeze(0), h_omic.unsqueeze(0)).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path, h_omic], dim=0))
        else:
            h = h_path + h_omic

        logits = self.classifier(h).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        attention_scores = {
            'coattn': A_coattn,
            'path': A_path,
            'omic': A_omic,
            'film_gamma': gamma,
            'film_beta': beta
        }

        return hazards, S, Y_hat, attention_scores