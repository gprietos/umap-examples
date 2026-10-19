"""Image embedding models, and embedding a FiftyOne dataset into a sample field."""

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


### ------------------------------------------------
### --- Models -------------------------------------
### ------------------------------------------------

# name -> (kind, model id)
MODELS = {
    "dinov2_vits14": ("timm", "vit_small_patch14_dinov2.lvd142m"),
    "dinov2_vitb14": ("timm", "vit_base_patch14_dinov2.lvd142m"),
    "dinov2_vitl14": ("timm", "vit_large_patch14_dinov2.lvd142m"),
    "clip_vitb32":   ("fiftyone", "clip-vit-base32-torch"),
}


class ResizeWithPadding:
    """Resize the longest side to `size` and pad to a square, keeping the aspect ratio."""

    def __init__(self, size, fill=0, interpolation=Image.BICUBIC):
        self.size = size
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = F.resize(img, (new_h, new_w), self.interpolation)
        pad_w = self.size - new_w
        pad_h = self.size - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return F.pad(img, padding, self.fill)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_timm_model(model_id, img_size=224):
    """Load a timm model (CLS-token embedding, no head) and its transforms.

    DINOv2 is pretrained at 518 px; img_size=224 (16x16 patches) is much faster and enough for global embeddings.
    """
    import timm

    model = timm.create_model(model_id, pretrained=True, num_classes=0, img_size=img_size)
    model = model.eval().to(get_device())

    data_config = timm.data.resolve_model_data_config(model)
    transforms = torchvision.transforms.Compose([
        ResizeWithPadding(img_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),
    ])
    return model, transforms


### ------------------------------------------------
### --- Embeddings ---------------------------------
### ------------------------------------------------

class _ImageFiles(Dataset):
    def __init__(self, filepaths, transforms):
        self.filepaths = filepaths
        self.transforms = transforms

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, i):
        return self.transforms(Image.open(self.filepaths[i]).convert("RGB"))


@torch.no_grad()
def embed_files(filepaths, model, transforms, batch_size=64, num_workers=4):
    loader = DataLoader(_ImageFiles(filepaths, transforms), batch_size=batch_size, num_workers=num_workers)
    device = get_device()
    out = []
    for batch in tqdm(loader, desc="Embedding"):
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(model(batch.to(device)).float().cpu().numpy())
    return np.concatenate(out)


def embed(dataset, model_name="dinov2_vitb14", field=None, batch_size=64, num_workers=2, overwrite=False):
    """Embed the samples of a FiftyOne dataset into the field `emb_<model_name>` and return the field name.

    Only samples without an embedding are embedded, so re-runs are a no-op. Pass overwrite=True after changing
    a model's preprocessing. Embeddings are stored raw (not normalized): UMAP's cosine metric ignores their length.
    Read them with `np.stack(dataset.values(field))`.
    Each DataLoader worker is a fork of the (large) Jupyter kernel, so keep num_workers low to avoid running out of RAM.
    """
    field = field or f"emb_{model_name}"
    view = dataset if overwrite or not dataset.has_field(field) else dataset.exists(field, False)
    if len(view) == 0:
        return field

    kind, model_id = MODELS[model_name]
    if kind == "timm":
        model, transforms = get_timm_model(model_id)
        X = embed_files(view.values("filepath"), model, transforms, batch_size=batch_size, num_workers=num_workers)
        view.set_values(field, list(X))
    else:
        import fiftyone.zoo as foz

        view.compute_embeddings(
            foz.load_zoo_model(model_id), embeddings_field=field,
            batch_size=batch_size, num_workers=num_workers, progress=True,
        )
    return field
