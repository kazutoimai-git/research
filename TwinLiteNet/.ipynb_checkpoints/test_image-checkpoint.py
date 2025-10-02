import torch
import numpy as np
import shutil
from tqdm.autonotebook import tqdm
import os
from model import TwinLite as net
import cv2

def Run(model, img):
    if img is None:
        raise ValueError("Input image is None")

    img = cv2.resize(img, (640, 360))
    img_rs = img.copy()

    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img, 0)  # add a batch dimension
    img = img.cuda().float() / 255.0

    with torch.no_grad():
        img_out = model(img)

    x0 = img_out[0]
    x1 = img_out[1]

    _, da_predict = torch.max(x0, 1)
    _, ll_predict = torch.max(x1, 1)

    DA = da_predict.byte().cpu().data.numpy()[0] * 255
    LL = ll_predict.byte().cpu().data.numpy()[0] * 255
    img_rs[DA > 100] = [255, 0, 0]
    img_rs[LL > 100] = [0, 255, 0]

    return img_rs


# モデル準備
model = net.TwinLiteNet()
model = torch.nn.DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load('pretrained/best.pth'))
model.eval()

# 結果保存フォルダの作成
if os.path.exists('results'):
    shutil.rmtree('results')
os.makedirs('results', exist_ok=True)

# 画像ファイルのみ取得
image_list = [f for f in os.listdir('images') if f.lower().endswith(('.jpg', '.png'))]

for i, imgName in enumerate(image_list):
    img_path = os.path.join('images', imgName)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Warning: Cannot read image {img_path}, skipped.")
        continue

    img_rs = Run(model, img)
    cv2.imwrite(os.path.join('results', imgName), img_rs)
