stage2_config = {
    "stage1_ckpt":  "/work/u4001296/project1/cell_cls/runs/stage1_v3/best.pt",
    "output_dir":   "/work/u4001296/project1/cell_cls/runs/stage2_v3",
    "crops_dir":    "/work/u4001296/project1/cell_cls/wsi_crops",

    "model_name":  "tf_efficientnetv2_m.in21k_ft_in1k",
    "num_classes": 17,
    "val_ratio":   0.2,

    "batch_size":  32,
    "epochs":      50,
    "lr":          1e-4,
    "weight_decay": 1e-2,
    "seed":        42,
    "amp":         True,
    "patience":    10,
    "focal_gamma": 2.0,
}

stage1_config = {
    "data_dir":    "/work/u4001296/project1/data/mll/bone_marrow_cell_dataset",
    "output_dir":  "/work/u4001296/project1/cell_cls/runs/stage1_v3",

    "model_name":  "tf_efficientnetv2_m.in21k_ft_in1k",
    "num_classes": 17,
    "img_size":    224,

    "batch_size":  64,
    "num_workers": 8,
    "epochs":      30,
    "lr":          1e-4,
    "weight_decay": 1e-2,
    "val_ratio":   0.1,
    "seed":        42,
    "amp":         True,
    "patience":    5,
    "focal_gamma": 2.0,
}