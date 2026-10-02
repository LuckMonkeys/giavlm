"""Deterministic bounded prefix search, separate from continuous optimization."""


def expansion_batch(beam, candidates, cursor, count):
    width = len(candidates)
    result = []
    for index in range(cursor, min(cursor + count, len(beam) * width)):
        parent = beam[index // width]
        result.append({"tokens": parent["tokens"] + [candidates[index % width]],
                       "score": parent["score"]})
    return result


def retain_best(existing, additions, width):
    return sorted(existing + additions,
                  key=lambda item: (item["score"], item["tokens"]))[:width]
