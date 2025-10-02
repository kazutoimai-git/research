import torch
import numpy as np
import cv2
import time
from model import TwinLite as net

def Run(model, img):
    # test_image.pyから流用した、1フレームを処理する関数
    img_rs = cv2.resize(img, (640, 360))
    
    img_tensor = img_rs[:, :, ::-1].transpose(2, 0, 1)
    img_tensor = np.ascontiguousarray(img_tensor)
    img_tensor = torch.from_numpy(img_tensor)
    img_tensor = torch.unsqueeze(img_tensor, 0)  # バッチ次元の追加
    img_tensor = img_tensor.cuda().float() / 255.0

    with torch.no_grad():
        img_out = model(img_tensor)
    
    x0 = img_out[0]
    x1 = img_out[1]

    _, da_predict = torch.max(x0, 1)
    _, ll_predict = torch.max(x1, 1)

    DA = da_predict.byte().cpu().data.numpy()[0] * 255
    LL = ll_predict.byte().cpu().data.numpy()[0] * 255
    
    # 走行可能領域を青色、車線を緑色で描画
    img_rs[DA > 100] = [255, 0, 0]
    img_rs[LL > 100] = [0, 255, 0]
    
    return img_rs


# --- メイン処理 ---
# 1. モデルの読み込み
model = net.TwinLiteNet()
model = torch.nn.DataParallel(model)
model = model.cuda()
model.load_state_dict(torch.load('pretrained/best.pth'))
model.eval()

# 2. 入力ビデオと出力ビデオの設定
input_video_path = 'data/video/rain_video.mp4'  # ★処理したい動画のパスに変更してください
output_video_path = 'results/video.mp4' # ★結果を保存する動画のパス

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print(f"エラー: {input_video_path} を開けませんでした。")
    exit()

# 出力ビデオの設定
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (640, 360)) # 出力サイズはモデルの入力に合わせる

# 3. FPS計測用変数
frame_count = 0
total_time = 0.0

# 4. 1フレームずつ処理するループ
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # 各フレームに対して推論を実行
    result_frame = Run(model, frame)

    end_time = time.time()
    elapsed = end_time - start_time
    fps_current = 1.0 / elapsed if elapsed > 0 else 0

    frame_count += 1
    total_time += elapsed

    print(f"Frame {frame_count}: {fps_current:.2f} FPS")

    # 結果を保存
    out.write(result_frame)

    # 'q'キーで処理を中断
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. 平均FPSを表示
avg_fps = frame_count / total_time if total_time > 0 else 0
print(f"平均FPS: {avg_fps:.2f}")

# 6. 終了処理
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"処理が完了しました。結果は {output_video_path} に保存されました。")
