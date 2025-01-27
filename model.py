import torch
import os
# from monai.networks.nets.segresnet_ds import SegResNetEncoder
from utils import adjust_prefix_and_load_state_dict
import streamlit as st
import monai
from lighter_zoo import SegResEncoder

# Wrap the segresnet model in a module that returns the embeddings
class EmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        with st.spinner("Downloading model..."):
            self.model = SegResEncoder.from_pretrained("project-lighter/ct_fm_feature_extractor")
        
        # import sys
        # sys.path.append('/home/suraj/Repositories/lighter-ct-fm')

        # from models.suprem import SuPreM_loader
        # from models.backbones.unet3d import UNet3D

        # self.model = SuPreM_loader(
        #     model=UNet3D(
        #         n_class=10
        #     ),
        #     ckpt_path="/mnt/data1/CT_FM/baselines/SuPreM_UNet/supervised_suprem_unet_2100.pth",
        #     decoder=False,
        #     encoder_only=True
        # )

        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = torch.nn.Flatten(start_dim=1)

    def forward(self, x):
        x = x.permute(0, 1, 4, 3, 2)
        x = x.flip(2).flip(3)
        x = self.model(x)
        if isinstance(x, list):
            x = x[-1]
        x = self.avgpool(x)
        x = self.flatten(x)
        return x
    
def load_model():
    model = EmbeddingModel()
    model.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.eval()
    return model
