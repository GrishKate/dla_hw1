import hydra
from src.metrics.utils import calc_cer, calc_wer


@hydra.main(version_base=None, config_path="src/configs", config_name="calc_cer_wer")
def main(config):
    target_path = config.target_path
    pred_path = config.pred_path
    with open(target_path, 'r') as f:
        target = f.read()
    with open(pred_path, 'r') as f:
        pred = f.read()
    wer = calc_wer(target, pred)
    cer = calc_cer(target, pred)
    return cer, wer


if __name__ == "__main__":
    main()
