import cv2
import torch
import numpy as np
import os
import sys
import argparse

# --- 1. モデルのリポジトリへのパスを追加 ---
# このスクリプト(pipeline.py)が 'research' ディレクトリにあり、
# 'Raindrop-Removal' と 'TwinLiteNet' も同じ 'research' ディレクトリにあると仮定した構成
# 'research/'
#  |- pipeline.py
#  |- Raindrop-Removal/
#  |- TwinLiteNet/

# スクリプトの場所を基準に各リポジトリへのパスを追加
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'Raindrop-Removal'))
sys.path.append(os.path.join(script_dir, 'TwinLiteNet'))


# --- 2. モデル定義とヘルパー関数の準備 ---

# ▼▼▼ 'hyukju/raindrop-removal' から必要なクラスと関数をインポート ▼▼▼
try:
    from model.proposed_model import Model as RainRemovalModel
    from utils import numpy2tensor, tensor2numpy
    print("Successfully imported Raindrop-Removal components.")
except ImportError as e:
    print(f"Failed to import Raindrop-Removal components, rain removal will be skipped. Error: {e}")
    RainRemovalModel = None
    # プレースホルダーとしてダミー関数を定義
    def numpy2tensor(nump, mean=0.5, std=0.5): pass
    def tensor2numpy(tensor, mean=0.5, std=0.5): pass


# ▼▼▼ 'chequanghuy/twinlitenet' から必要なクラスをインポート ▼▼▼
try:
    from model.TwinLite import TwinLiteNet
    print("Successfully imported TwinLiteNet components.")
except ImportError as e:
    print(f"Failed to import TwinLiteNet components, lane detection will be skipped. Error: {e}")
    TwinLiteNet = None


# --- 3. パイプライン処理のメイン関数 ---

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
    rain_removal_model = None
    if RainRemovalModel:
        try:
            rain_removal_model = RainRemovalModel(phase='test')
            # スクリプトからの相対パスで重みファイルへのパスを指定
            rain_weights_path = os.path.join(script_dir, 'Raindrop-Removal', 'checkpoint', 'proposed', 'model_epoch1800.pth')

            if os.path.exists(rain_weights_path):
                # loadメソッドにはディレクトリパスを渡す必要があるかもしれません。
                # もしエラーが出る場合は、元のリポジトリの仕様をご確認ください。
                rain_removal_model.load(os.path.dirname(rain_weights_path))
                rain_removal_model.generator.to(device)
                rain_removal_model.generator_mask.to(device)
                rain_removal_model.generator.eval()
                rain_removal_model.generator_mask.eval()
                print("Rain removal model loaded successfully.")
            else:
                print(f"Warning: Rain removal weights not found at {rain_weights_path}. Rain removal will be skipped.")
                rain_removal_model = None

        except Exception as e:
            print(f"An error occurred while loading the RainRemovalModel: {e}")
            rain_removal_model = None
    else:
        print("RainRemovalModel not imported. Skipping rain removal model loading.")


    # 2. 白線検出モデル (TwinLiteNet) のロード
    lane_detection_model = None
    if TwinLiteNet:
        try:
            lane_detection_model = TwinLiteNet()
            lane_detection_model = torch.nn.DataParallel(lane_detection_model)
            lane_detection_model = lane_detection_model.to(device)
            # スクリプトからの相対パスで重みファイルへのパスを指定
            lane_weights_path = os.path.join(script_dir, 'TwinLiteNet', 'pretrained', 'best.pth')

            if not os.path.exists(lane_weights_path):
                raise FileNotFoundError(f"Lane detection weights not found at {lane_weights_path}")

            lane_detection_model.load_state_dict(torch.load(lane_weights_path, map_location=device))
            lane_detection_model.eval()
            print("Lane detection model loaded successfully.")
        except Exception as e:
            print(f"An error occurred while loading the TwinLiteNet model: {e}")
            lane_detection_model = None
    else:
        print("TwinLiteNet not imported. Skipping lane detection model loading.")


    # --- 動画の入出力設定 ---
    if not os.path.exists(input_video_path):
        print(f"Error: Could not open video file {input_video_path}")
        return

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Failed to open video capture for {input_video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

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
        processed_frame = frame

        # --- ステップ1: 雨除去 ---
        if rain_removal_model:
            input_tensor = numpy2tensor(frame_rgb).to(device)

            with torch.no_grad():
                mask  = rain_removal_model.generator_mask(input_tensor)
                output_tensor, _, _ = rain_removal_model.generator(torch.cat((input_tensor, mask), dim=1))

            rain_removed_frame_np = tensor2numpy(output_tensor).squeeze(0)
            rain_removed_frame = (rain_removed_frame_np * 255).astype(np.uint8)
            processed_frame = cv2.cvtColor(rain_removed_frame, cv2.COLOR_RGB_BGR)

        # --- ステップ2: 白線検出 ---
        if lane_detection_model:
            img_for_lane = cv2.resize(processed_frame, (640, 360))
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

            img_rs[DA > 100] = [255, 0, 0]
            img_rs[LL > 100] = [0, 255, 0]

            processed_frame = cv2.resize(img_rs, (frame_width, frame_height))

        out.write(processed_frame)

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")

    # --- 処理完了 ---
    print(f"Processing complete. Output saved to {output_video_path}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rain removal and lane detection pipeline for driving videos.')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input rainy video file.')
    # 出力先のデフォルト値を変更
    parser.add_argument('--output', type=str, default='results/result.mp4',
                        help='Path to save the processed output video.')
    args = parser.parse_args()

    main_pipeline(args.input, args.output)