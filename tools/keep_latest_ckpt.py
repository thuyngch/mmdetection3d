import numpy as np
import os, argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=str, help="Directory")
args = parser.parse_args()

latest_pth = os.path.join(args.dir, "latest.pth")
if os.path.exists(latest_pth):
    os.remove(latest_pth)

files = sorted(glob(os.path.join(args.dir, "*.pth")))
print("Number of files:", len(files))

if len(files):
    epochs = np.array([int(file.split('_')[-1].split('.')[0]) for file in files])
    max_epoch_idx = np.argmax(epochs)
    print(max_epoch_idx)

    for idx, file in enumerate(files):
        if idx == max_epoch_idx:
            continue
        os.remove(file)
        print("Remove file \"{}\"".format(file))
