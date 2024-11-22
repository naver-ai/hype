import torch
import math
import os

ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference.pt")
ref = torch.load(ref_path)
img_ref, txt_ref = ref["img"], ref["txt"]


@torch.cuda.amp.autocast(enabled=False)
def expm(v, curvature, time_keepdim=False):
    v, curvature = v.float(), curvature.float()
    x_space_temp = torch.sqrt(curvature) * torch.norm(v, dim=-1, keepdim=True)
    x_space = (
        torch.sinh(torch.clamp(x_space_temp, min=1e-8, max=math.asinh(2**15))) * v / torch.clamp(x_space_temp, min=1e-8)
    )
    x_time = torch.sqrt(1 / curvature + torch.sum(x_space**2, dim=-1, keepdim=time_keepdim))  # [B, D]
    return x_space, x_time


@torch.cuda.amp.autocast(enabled=False)
def similarity(x, y, curvature):
    x, y = x.float(), y.float()
    curvature = curvature.float()
    x_space, x_time = expm(x, curvature, time_keepdim=True)
    y_space, y_time = expm(y, curvature, time_keepdim=True)
    xy_inner = x_space @ y_space.T - x_time * y_time.T
    lorentzian_distance = torch.rsqrt(curvature) * torch.acosh(torch.clamp(-curvature * xy_inner, min=1e-8))
    return -lorentzian_distance


@torch.no_grad()
def entailment(x, y, curvature):
    x_space, x_time = expm(x, curvature, time_keepdim=True)
    y_space, y_time = expm(y, curvature, time_keepdim=True)

    K = 0.1
    x_euc_norm = torch.norm(x_space, dim=-1, keepdim=True)
    denominator = torch.sqrt(curvature) * x_euc_norm + 1e-8
    aperture_x = torch.arcsin(torch.clamp(2 * K / denominator, -1 + 1e-8, 1 - 1e-8))

    xy_inner = x_space @ y_space.T - x_time * y_time.T
    denominator = x_euc_norm * torch.sqrt(torch.clamp((curvature * xy_inner) ** 2 - 1, min=1e-8)) + 1e-8
    numerator = y_time.T + x_time * curvature * xy_inner
    exterior_xy = torch.arccos(torch.clamp(numerator / denominator, -1.0 + 1e-8, 1.0 - 1e-8))

    return exterior_xy - aperture_x


def specificity(image=None, text=None, curv=None):
    assert (image is not None) ^ (text is not None), "Either image or text must be provided but not both"
    assert curv is not None, "Curvature must be provided"

    global img_ref, txt_ref

    if image is not None:
        txt_ref = txt_ref.to(image.device)
        ient = entailment(txt_ref, image, curv)
        return ient.mean(dim=0)
    else:
        img_ref = img_ref.to(text.device)
        tent = entailment(text, img_ref, curv)
        return tent.mean(dim=1)
