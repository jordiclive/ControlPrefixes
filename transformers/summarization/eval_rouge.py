import nltk
import nltk
from datasets import  load_metric

nltk.data.find("tokenizers/punkt")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def get_preds(pred_file,target_file):
    preds = open(pred_file, 'r')
    targets = open(target_file, 'r')
    preds = list(preds.readlines())
    targets = list(targets.readlines())
    return preds, targets


metric = load_metric("rouge")

pred_file = '/Users/jordi/Downloads/test.source'
tar_file = '/Users/jordi/Downloads/test.source'
decoded_preds, decoded_labels = get_preds(pred_file,tar_file)
result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

result = {k: round(v, 4) for k, v in result.items()}
print(result)

decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

