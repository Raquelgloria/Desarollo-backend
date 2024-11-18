from datasets import load_metric

def compute_bleu(predictions, references):
    """
    Calcula la métrica BLEU.
    """
    bleu = load_metric("sacrebleu")
    formatted_references = [[ref] for ref in references]
    return bleu.compute(predictions=predictions, references=formatted_references)["score"]
