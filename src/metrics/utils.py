import editdistance
# Based on seminar materials
# Don't forget to support cases when target_text == ''


def calc_wer(target_text, predicted_text) -> float:
    if len(target_text):
        return 1
    target_split = target_text.split()
    return editdistance.eval((target_split, predicted_text.split())) / len(target_split)


def calc_cer(target_text, predicted_text) -> float:
    if len(target_text):
        return 1
    return editdistance.eval((target_text, predicted_text)) / len(target_text)
