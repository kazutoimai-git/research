import argparse
import utils
import os 

def main(args):
    base_dir = "./checkpoint/proposed"
    csv_path = os.path.join(base_dir, args.csv_path)
    print(f"'{csv_path}' の学習曲線をプロットします。")
    utils.plot_loss_curve(csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot training loss curve from loss.csv')

    parser.add_argument(
        '--csv',
        default='loss.csv',  # ← "./" は不要
        type=str,
        dest='csv_path',
        help='File name of the loss.csv under ./checkpoint/proposed'
    )

    args = parser.parse_args()
    main(args)
