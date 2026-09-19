"""Competition metric: MRR@25 with InChIKey14 matching.

The official matcher passes both prediction and answer through RDKit tautomer
canonicalization (pinned at 2026.03.3) and compares the first block of the
InChIKey. Stereochemistry and tautomer form are therefore free.

Everything here is deliberately cheap to call in a loop: canonicalization is
slow (tens of ms for a big molecule) so it is cached hard.
"""
from __future__ import annotations

import functools
from typing import Iterable, Mapping, Sequence

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

MAX_GUESSES = 25

_TAUTOMER = None


def _tautomer_enumerator():
    global _TAUTOMER
    if _TAUTOMER is None:
        params = rdMolStandardize.CleanupParameters()
        _TAUTOMER = rdMolStandardize.TautomerEnumerator(params)
    return _TAUTOMER


@functools.lru_cache(maxsize=1_000_000)
def inchikey14(smiles: str) -> str | None:
    """Canonical 2D identity of a SMILES, or None if it will not parse.

    None means the string is not a usable prediction. Never silently turn an
    unparseable SMILES into a match -- the grader would score it zero and you
    want your own CV to agree with the grader.
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = _tautomer_enumerator().Canonicalize(mol)
    except Exception:
        pass  # keep the uncanonicalized molecule rather than dropping the guess
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return None
    if not key:
        return None
    return key.split("-")[0]


def reciprocal_rank(guesses: Sequence[str], answer: str, k: int = MAX_GUESSES) -> float:
    """1/rank of the first guess whose InChIKey14 matches, else 0.0."""
    target = inchikey14(answer)
    if target is None:
        raise ValueError(f"answer SMILES did not parse: {answer!r}")
    for i, g in enumerate(guesses[:k], start=1):
        if inchikey14(g) == target:
            return 1.0 / i
    return 0.0


def mrr_at_k(
    predictions: Mapping[str, Sequence[str]],
    answers: Mapping[str, str],
    k: int = MAX_GUESSES,
) -> float:
    """Mean reciprocal rank over molecules. Missing molecule_id scores 0."""
    if not answers:
        return 0.0
    total = 0.0
    for mol_id, answer in answers.items():
        total += reciprocal_rank(predictions.get(mol_id, ()), answer, k=k)
    return total / len(answers)


def per_molecule_rr(
    predictions: Mapping[str, Sequence[str]],
    answers: Mapping[str, str],
    k: int = MAX_GUESSES,
) -> dict[str, float]:
    """Reciprocal rank per molecule -- use this to slice by novelty class."""
    return {
        mol_id: reciprocal_rank(predictions.get(mol_id, ()), ans, k=k)
        for mol_id, ans in answers.items()
    }


class SubmissionError(ValueError):
    pass


def validate_submission(predictions: Mapping[str, Sequence[str]],
                        expected_ids: Iterable[str] | None = None) -> None:
    """Raise on anything the grader would reject.

    The grader rejects a submission that is missing a column, is empty, has
    nulls, repeats a molecule_id, or gives more than 25 guesses for any
    molecule. Duplicate ids cannot happen in a dict, so the checks that matter
    here are coverage, emptiness, nulls and cardinality.
    """
    if not predictions:
        raise SubmissionError("submission is empty")
    for mol_id, guesses in predictions.items():
        if mol_id is None or mol_id == "":
            raise SubmissionError("null or empty molecule_id")
        if guesses is None or len(guesses) == 0:
            raise SubmissionError(f"{mol_id}: no guesses -- the smiles field would be null")
        if len(guesses) > MAX_GUESSES:
            raise SubmissionError(f"{mol_id}: {len(guesses)} guesses, max is {MAX_GUESSES}")
        for g in guesses:
            if not g or ";" in g:
                raise SubmissionError(f"{mol_id}: bad SMILES {g!r} (empty or contains ';')")
    if expected_ids is not None:
        expected = set(expected_ids)
        got = set(predictions)
        missing = expected - got
        extra = got - expected
        if missing:
            raise SubmissionError(f"{len(missing)} molecule_ids missing, e.g. {sorted(missing)[:3]}")
        if extra:
            raise SubmissionError(f"{len(extra)} unexpected molecule_ids, e.g. {sorted(extra)[:3]}")


def write_submission(predictions: Mapping[str, Sequence[str]], path: str,
                     expected_ids: Iterable[str] | None = None) -> str:
    """Validate then write submission.csv. Validation first, always."""
    validate_submission(predictions, expected_ids)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("molecule_id,smiles\n")
        for mol_id in sorted(predictions):
            fh.write(f"{mol_id},{';'.join(predictions[mol_id])}\n")
    return path
