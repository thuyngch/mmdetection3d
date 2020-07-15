import numpy as np
import mmcv, torch, os, argparse

from mmdet3d.datasets import build_dataset
from mmdet3d.apis import show_result_meshlab

# ArgumentParser
parser = argparse.ArgumentParser()

parser.add_argument("--cfg", type=str, default="configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py",
                    help="Config file")

parser.add_argument("--idx", type=int, default=0, help="Sample index in the dataset")

parser.add_argument("--outdir", type=int, default='cache/', help="Sample index in the dataset")

args = parser.parse_args()

# Build dataset
cfg = mmcv.Config.fromfile(args.cfg)
dataset = build_dataset(cfg.data.train)

# Select a sample
sample = dataset.__getitem__(args.idx)

# Convert to MeshLab
data = dict(
    img_metas=[[sample['img_metas'].data]],
    points=[[sample['points'].data]],
)
gts = dict(
    boxes_3d=sample['gt_bboxes_3d'].data,
    scores_3d=torch.ones([len(sample['gt_labels_3d'].data)]),
    labels_3d=sample['gt_labels_3d'].data,
)

os.makedirs(args.outdir, exist_ok=True)
show_result_meshlab(data, gts, args.outdir)
print("\nVisualization result is saved in \'{}\'. Please use MeshLab to visualize it.".format(args.outdir))
