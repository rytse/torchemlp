def dictify_rep(rep):
    return tuple(
        [
            (k, v)
            for k, v in rep.__dict__.items()
            if (k not in ["size", "is_permutation", "is_orthogonal"])
        ]
    )


def mul_reps(rep1, rep2):
    if isinstance(rep1, int):
        ra = rep1
        rb = rep2
    elif isinstance(rep2, int):
        ra = rep2
        rb = rep1
    else:
        raise ValueError("mul_reps: at least one rep must be an int")

    if rb == 1:
        return ra
    if rb == 0:
        return 0

    if ra.concrete:
        return SumRep(*(rb * [ra]))
    return DeferredSumRep(*(rb * [ra]))
