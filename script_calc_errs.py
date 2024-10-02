from src.metrics.utils import calc_cer, calc_wer


def calc_cer_wer(target_path, pred_path):
    with open(target_path, 'r') as f:
        target = f.read()
    with open(pred_path, 'r') as f:
        pred = f.read()
    wer = calc_wer(target, pred)
    cer = calc_cer(target, pred)
    return cer, wer
