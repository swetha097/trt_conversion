import nltk
from nltk.translate.bleu_score import sentence_bleu
# from pycocoevalcap.meteor.meteor import Meteor
from nltk.tokenize import word_tokenize

from rouge_score import rouge_scorer
from sklearn.metrics import precision_recall_fscore_support

# Example Ground Truth and Generated Reports
gt_report = "The cat sat on the mat"
generated_report = "The cat is sitting on the mat"

# Tokenize the reports (for BLEU)
gt_tokens = gt_report.split()
generated_tokens = generated_report.split()

# BLEU Scores (1-gram to 4-gram)
def calculate_bleu_scores(gt_report, generated_report):
    bleu_1 = sentence_bleu([gt_report.split()], generated_report.split(), weights=(1, 0, 0, 0))
    bleu_2 = sentence_bleu([gt_report.split()], generated_report.split(), weights=(0.5, 0.5, 0, 0))
    bleu_3 = sentence_bleu([gt_report.split()], generated_report.split(), weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = sentence_bleu([gt_report.split()], generated_report.split(), weights=(0.25, 0.25, 0.25, 0.25))
    
    return bleu_1, bleu_2, bleu_3, bleu_4

# METEOR Score
def calculate_meteor_score(gt_report, generated_report):
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('wordnet')
    # return nltk.translate.meteor_score.meteor_score([word_tokenize(gt_report)], [word_tokenize(generated_report)])
    return nltk.translate.meteor_score.meteor_score([word_tokenize(gt_report)], word_tokenize(generated_report))

# ROUGE-L Score
def calculate_rouge_l(gt_report, generated_report):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = scorer.score(gt_report, generated_report)
    return score["rougeL"].fmeasure

# Classification Metrics (Precision, Recall, F1)
def calculate_classification_metrics(gt_report, generated_report):
    gt_tokens = set(gt_report.split())
    generated_tokens = set(generated_report.split())
    
    # True positives, false positives, false negatives
    tp = len(gt_tokens & generated_tokens)
    fp = len(generated_tokens - gt_tokens)
    fn = len(gt_tokens - generated_tokens)
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return precision, recall, f1


list_gt = ["Image with pneumonia bacteria", 
           "Image with pneumonia virus",
           "Image with pneumonia virus",
           "Lungs are hyperexpanded. There is no focal airspace consolidation. No suspicious pulmonary mass or nodule is seen.No pleural effusion or pneumothorax. Normal heart size and mediastinal contour.",
           "The heart size and pulmonary vascularity appear within normal limits. A large hiatal hernia is noted. The lungs are free of focal airspace disease. No pneumothorax or pleural effusion is seen. Degenerative changes are present in the spine.,",
           "The heart size is upper limits of normal. The pulmonary XXXX and mediastinum are within normal limits. There is no pleural effusion or pneumothorax. There is no focal air space opacity to suggest a pneumonia.",
           "The outside x-XXXX is normal except for slight cardiomegaly.",
           "Heart size is normal. The aorta is tortuous, and cannot exclude ascending aortic aneurysm. The pulmonary vascularity is normal. There residual to prior granulomatous infection. Lungs are otherwise clear. Degenerative change of the spine.",
           "Heart size is normal. The aorta is tortuous, and cannot exclude ascending aortic aneurysm. The pulmonary vascularity is normal. There residual to prior granulomatous infection. Lungs are otherwise clear. Degenerative change of the spine."
           ]
list_generated_report = ["Based on the image, the condition depicted is 'pneumonia.' The presence of a white spot in the lungs suggests an infection or inflammation, which is a common symptom of pneumonia. The image also shows the lungs to be enlarged, which could indicate the severity of the infection or inflammation. The radiology report would likely state that the patient has a pneumonia infection in the lungs, which requires further medical attention and treatment", 
                         "The image shows a baby's chest with a visible rib cage and a small amount of lung tissue. The rib cage is clearly visible, and the baby's chest is empty, with no signs of any medical conditions such as cardiomegaly, lung opacity, lung lesion, edema, pneumonia, atelectasis, pneumothorax, pleural effusion, pleural other, fractured, enlarged cardiomediastinum, or no finding.",
                         "The image shows a baby's chest with a visible rib cage and a small amount of lung tissue. The rib cage is clearly visible, and the baby's chest is empty, with no signs of any medical conditions such as cardiomegaly, lung opacity, lung lesion, edema, pneumonia, atelectasis, pneumothorax, pleural effusion, pleural other, fractured, enlarged cardiomediastinum, or no finding.",
                         "The image shows a chest X-ray of a person, which displays a lung lesion. The presence of a lung lesion suggests that the person might have a medical condition such as lung cancer or a benign lung tumor. However, without further information, it is not possible to determine the exact condition or diagnose the patient.",
                         "The image shows a chest X-ray of a person, which displays a lung lesion. The presence of the lung lesion suggests that the person might have a medical condition such as lung cancer or a benign lung tumor. The image does not show any other conditions like cardiomegaly, atelectasis, pneumonia, or pleural effusion.",
                         "The image shows a chest X-ray of a person, which displays a lung lesion. The presence of the lung lesion suggests that the person might have a medical condition such as lung cancer or a benign lung tumor. The image does not show any other conditions like cardiomegaly, atelectasis, pneumonia, or pleural effusion.",
                         "The image shows a person with a large chest, which could be indicative of a condition such as cardiomegaly.\n\nHowever, the presence of a large chest does not necessarily mean that the person has any of the other conditions listed. To accurately classify the conditions, a more detailed examination of the image would be required.",
                         "The image shows a chest X-ray of a person, which displays a lung lesion. The presence of a lung lesion suggests that the person might have a medical condition such as lung cancer, pulmonary fibrosis, or chronic obstructive pulmonary disease (COPD). However, without more information, it is not possible to definitively diagnose the specific condition.",
                         "The image shows a back view of a person's body, with a focus on the spine and ribcage. There are no visible signs of cardiomegaly, lung opacity, lung lesion, edema, pneumonia, atelectasis, pneumothorax, pleural effusion, pleural other, fractured, enlarged cardiomediastinum, or no finding."
                        ]
for gt_report, generated_report in zip(list_gt, list_generated_report):
    # Calculate all metrics
    bleu_1, bleu_2, bleu_3, bleu_4 = calculate_bleu_scores(gt_report, generated_report)
    meteor_score = calculate_meteor_score(gt_report, generated_report)
    rouge_l_score = calculate_rouge_l(gt_report, generated_report)
    ce_precision, ce_recall, ce_f1 = calculate_classification_metrics(gt_report, generated_report)
    # Displaying the results
    print(f"test_BLEU_1    : {bleu_1}")
    print(f"test_BLEU_2    : {bleu_2}")
    print(f"test_BLEU_3    : {bleu_3}")
    print(f"test_BLEU_4    : {bleu_4}")
    print(f"test_METEOR    : {meteor_score}")
    print(f"test_ROUGE_L   : {rouge_l_score}")
    print(f"test_ce_precision: {ce_precision}")
    print(f"test_ce_recall : {ce_recall}")
    print(f"test_ce_f1     : {ce_f1}")






