def dictify_rep(rep):
    return tuple(
        [
            (k, v)
            for k, v in rep.__dict__.items()
            if (k not in ["size", "is_permutation", "is_orthogonal"])
        ]
    )
