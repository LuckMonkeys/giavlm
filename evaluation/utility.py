import re


def normalize_answer(text):
    """Documented lightweight normalization; not the official VQA punctuation script."""
    numbers = dict(zip("zero one two three four five six seven eight nine ten".split(), map(str, range(11))))
    tokens = re.findall(r"\w+(?:'\w+)?", text.lower())
    return " ".join(numbers.get(word, word) for word in tokens if word not in {"a", "an", "the"})


def vqa_soft_accuracy(prediction, answers):
    if not answers:
        raise ValueError("VQA utility requires reference answers")
    prediction = normalize_answer(prediction)
    answers = [normalize_answer(answer) for answer in answers]
    if len(answers) == 1:
        return float(prediction == answers[0])
    return sum(min(1.0, sum(prediction == answer for j, answer in enumerate(answers) if j != i) / 3)
               for i in range(len(answers))) / len(answers)


def evaluate_utility(adapter, rows, batch_size=4):
    from core.data import load_batch
    predictions = []
    for start in range(0, len(rows), batch_size):
        batch = load_batch(rows[start:start + batch_size], adapter)
        predictions.extend(adapter.generate(batch.images, batch.questions))
    if adapter.training_spec.task == "vqa":
        scores = [vqa_soft_accuracy(p, row["references"]) for row, p in zip(rows, predictions)]
        metrics = {"vqa_soft_accuracy": sum(scores) / max(1, len(scores))}
        policy = "leave-one-reference-out consensus; lowercase, word punctuation, articles and numbers"
    else:
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        tokenizer = PTBTokenizer()
        references = {str(i): [{"caption": r} for r in row["references"]] for i, row in enumerate(rows)}
        hypotheses = {str(i): [{"caption": p}] for i, p in enumerate(predictions)}
        score, _ = Cider().compute_score(tokenizer.tokenize(references), tokenizer.tokenize(hypotheses))
        metrics, policy = {"cider": float(score)}, "pycocoevalcap CIDEr with PTBTokenizer (requires Java)"
    return {"metrics": metrics, "normalization": policy,
            "predictions": [{"sample_id": row["sample_id"], "prediction": prediction}
                            for row, prediction in zip(rows, predictions)]}
