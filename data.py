import monai
import tempfile
import streamlit as st

def load_scan(path):
    """
    Load and preprocess a CT scan from a file path or uploaded file.

    Args:
        path (str or UploadedFile): The file path or uploaded file object of the CT scan.

    Returns:
        dict: A dictionary containing the preprocessed CT scan image tensor with key "image".
              Returns None if the input path is None.
    """
    if path is None:
        return None
    
    # Define the preprocessing transforms
    transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=["image"], ensure_channel_first=True),
        monai.transforms.EnsureTyped(keys=["image"]),
        monai.transforms.Orientationd(keys=["image"], axcodes="SPL"),
        # monai.transforms.Orientationd(keys=["image"], axcodes="ras"),
        monai.transforms.Spacingd(keys=["image"], pixdim=[3,1,1], mode="bilinear"),
        monai.transforms.CropForegroundd(keys=["image"], source_key="image"),
        monai.transforms.ScaleIntensityRanged(keys="image", a_min=-1024, a_max=2048, b_min=0, b_max=1, clip=True),
        monai.transforms.Lambda(func=lambda x: x["image"].as_tensor())
    ])

    # Load and preprocess the scan
    if isinstance(path, str):
        # If path is a string, assume it's a file path
        data = {"image": path}
        image = transforms(data)
    else:
        # If path is not a string, assume it's an uploaded file object
        bytes_data = path.read()
        with tempfile.NamedTemporaryFile(suffix='.nii.gz') as tmp:
            tmp.write(bytes_data)
            tmp.seek(0)
            data = {"image": tmp.name}
            image = transforms(data)

    # Return the preprocessed image tensor in a dictionary
    return {"image": image}
