import cv2
import torch
import numpy as np
import os

# --- 1. モデル定義とヘルパー関数の準備 ---

# ▼▼▼ 'hyukju/raindrop-removal' から必要なクラスと関数をインポート ▼▼▼
# (実際の使用時は、これらのファイルを適切にインポートまたは同じファイル内に定義してください)
# from model.proposed_model import Model as RainRemovalModel
# from utils import numpy2tensor, tensor2numpy
# 例として、必要なクラスをこのファイル内に直接定義します。
# 実際のコードではファイルが分かれているため、import文を使用してください。
# --- ここに 'layers.py', 'networks.py', 'model/proposed_model.py' の内容を結合 ---
# (コードが長大になるため、このサンプルでは概念的な呼び出しに留めます)

# ▼▼▼ 'chequanghuy/twinlitenet' から必要なクラスをインポート ▼▼▼
from chequanghuy.twinlitenet.TwinLiteNet-524141821c373c886443cf4254e3e87787c2c55e.model.TwinLite import TwinLiteNet

# --- ユーティリティ関数 (raindrop-removal/utils.py より) ---
def numpy2tensor(nump, mean=0.5, std=0.5):
    if nump.dtype == np.uint8:
        nump = (nump / 255.0)
    nump = (nump - mean) / std
    if nump.ndim == 3:
        nump = nump[np.newaxis, ...]
    tensor = torch.from_numpy(nump.transpose(0, 3, 1, 2).astype('float32'))
    return tensor

def tensor2numpy(tensor, mean=0.5, std=0.5):
    arr = tensor.detach().cpu().numpy().transpose(0, 2, 3, 1)
    arr = arr * std + mean
    arr = np.clip(arr, 0, 1)
    return arr

# --- 2. パイプライン処理のメイン関数 ---

def main_pipeline(input_video_path, output_video_path):
    """
    雨除去と白線検出のパイプライン処理を実行するメイン関数
    """
    # --- デバイスの設定 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- モデルのロード ---
    print("Loading models...")
    # 1. 雨除去モデルのロード
    # find_modelの代わりに直接モデルをインスタンス化
    # 注意: 実際の 'proposed_model.py' は 'networks' や 'layers' に依存しています。
    # ここでは仮のモデルとして扱います。
    try:
        from hyukju.raindrop-removal.Raindrop-Removal-d3a60713a2462799b5f9815668fda9aaa76dea9c.model.proposed_model import Model as RainRemovalModel
        rain_removal_model = RainRemovalModel(phase='test')
        # README.md に記載の重みファイルをロード
        rain_weights_path = 'srv11/Raindrop-Removal/checkpoint/proposed/model_epoch1800.pth'
        8fda9aaa76dea9c/checkpoint/proposed/model_epoch1800.pth'
        if os.path.exists(rain_weights_path):
            rain_removal_model.load(os.path.dirname(rain_weights_path))
            rain_removal_model.generator.eval()
            rain_removal_model.generator_mask.eval()
            print("Rain removal model loaded successfully.")
        else:
            print(f"Warning: Rain removal weights not found at {rain_weights_path}. Rain removal will be skipped.")
            rain_removal_model = None

    except ImportError as e:
        print(f"Could not import RainRemovalModel, skipping rain removal. Error: {e}")
        rain_removal_model = None


    # 2. 白線検出モデル (TwinLiteNet) のロード
    lane_detection_model = TwinLiteNet()
    lane_detection_model = torch.nn.DataParallel(lane_detection_model)
    lane_detection_model = lane_detection_model.to(device)
    lane_weights_path = 'chequanghuy/twinlitenet/TwinLiteNet-524141821c373c886443cf4254e3e87787c2c55e/pretrained/best.pth'
    lane_detection_model.load_state_dict(torch.load(lane_weights_path, map_location=device))
    lane_detection_model.eval()
    print("Lane detection model loaded successfully.")


    # --- 動画の入出力設定 ---
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    print("Starting video processing pipeline...")
    frame_count = 0
    # --- フレームごとの処理ループ ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- ステップ1: 雨除去 ---
        if rain_removal_model:
            input_tensor = numpy2tensor(frame_rgb).to(device)
            
            with torch.no_grad():
                # test_one_imageを参考に推論部分を実装
                mask  = rain_removal_model.generator_mask(input_tensor)
                output_tensor, _, _ = rain_removal_model.generator(torch.cat((input_tensor, mask), dim=1))

            rain_removed_frame_np = tensor2numpy(output_tensor).squeeze(0)
            # スケールを0-255に戻し、BGR形式に変換
            rain_removed_frame = (rain_removed_frame_np * 255).astype(np.uint8)
            rain_removed_frame = cv2.cvtColor(rain_removed_frame, cv2.COLOR_RGB2BGR)
        else:
            # モデルがロードできなかった場合はスキップ
            rain_removed_frame = frame

        # --- ステップ2: 白線検出 ---
        # test_image.pyのRun関数を参考
        img_for_lane = cv2.resize(rain_removed_frame, (640, 360))
        img_rs = img_for_lane.copy()

        img_tensor = img_for_lane[:, :, ::-1].transpose(2, 0, 1)
        img_tensor = np.ascontiguousarray(img_tensor)
        img_tensor = torch.from_numpy(img_tensor)
        img_tensor = torch.unsqueeze(img_tensor, 0).to(device).float() / 255.0

        with torch.no_grad():
            output_da, output_ll = lane_detection_model(img_tensor)
        
        _, da_predict = torch.max(output_da, 1)
        _, ll_predict = torch.max(output_ll, 1)

        DA = da_predict.byte().cpu().data.numpy()[0] * 255
        LL = ll_predict.byte().cpu().data.numpy()[0] * 255
        
        # --- ステップ3: 結果の描画 ---
        # 走行可能領域を青色で描画
        img_rs[DA > 100] = [255, 0, 0]
        # 白線を緑色で描画
        img_rs[LL > 100] = [0, 255, 0]

        # 元のフレームサイズに戻す
        final_frame = cv2.resize(img_rs, (frame_width, frame_height))

        # --- ステップ4: 動画への書き込み ---
        out.write(final_frame)
        
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames...")

    # --- 処理完了 ---
    print(f"Processing complete. Output saved to {output_video_path}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # --- 設定項目 ---
    INPUT_VIDEO = 'TwinLiteNet/data/video/rain_video.mp4'  # <<< 入力する雨天動画のパスを指定
    OUTPUT_VIDEO = 'TwinLiteNet/results/video.mp4' # <<< 出力する動画のパスを指定
    
    # 処理を実行
    main_pipeline(INPUT_VIDEO, OUTPUT_VIDEO)