# The label-to-gas mapping

Decided 2026-09-23. **Changing this changes no measurement.** Training and
evaluation use the integer labels only; the mapping affects gas names in
documents and figures, and the chemical reasoning built on them.

## Adopted

```
1 = Ethanol   2 = Ammonia   3 = Ethylene
4 = Acetaldehyde   5 = Acetone   6 = Toluene
```

Batch 1 counts: 90 / 98 / 83 / **30** / 70 / 74.

## The three claims that had to be reconciled

**A. The dataset's own documentation** states the encoding directly:
"1: Ethanol; 2: Ethylene; 3: Ammonia; 4: Acetaldehyde; 5: Acetone; 6: Toluene",
and illustrates it with a row that is literally the first line of `batch1.dat`.

**B. The paper's Table 2** (copying the dataset's per-batch table) lists for
Batch 1: Ethanol 83, Ethylene 30, Ammonia 70, Acetaldehyde 98, Acetone 90,
Toluene 74. Read in column order 1-6, that requires label 1 to hold 83 rows. It
holds 90.

**C. Batch 1's principal-component structure** against the paper's Fig. 4(a),
which plots the same data coloured by gas name.

A and B cannot both be read in column order. The previous project, which had no
copy of the documentation (`exp/a3-confound-ablation:docs/dataset-inspection.md`
records "no dataset documentation was found"), resolved the conflict by matching
counts, adopting `1=Acetone, 2=Acetaldehyde, 3=Ethanol, 4=Ethylene, 5=Ammonia,
6=Toluene` and overriding A. This document reverses that.

## Why C decided it

The mapping was read off `reports/figures/batch1_pca.png`, drawn with label
integers only and no gas names so the figure could not lead the reading, and
compared against Fig. 4(a) by shape. The reading agreed with A on labels 1, 4, 5
and 6 and exchanged 2 with 3. Labels 2 and 3 overlap heavily in PC1-PC2, so
`reports/figures/batch1_pca_zoom23.png` isolates that pair, where PC3-PC4
separates them cleanly.

## The corroboration that was not used to decide, and is strong

Under the adopted mapping the two projects and the paper agree in a way they did
not before:

| | this project | paper CDCNN (Fig. S3) |
|---|---|---|
| dead class | **Acetaldehyde, recall 0.000** | **Acetaldehyde, recall 0.00** |
| where it goes | 37.7 % Ethanol, 59.3 % Toluene | 100 % Ethanol |

Under the previous mapping the same numbers read as "this project loses Ethylene
while the paper loses Acetaldehyde" - two unrelated failures on opposite classes.
Three models failing on one class and sending it to one place is a far more
economical reading of the same numbers.

This corroboration is about agreement between models, not about chemistry. An
earlier draft claimed Acetaldehyde and Ethanol are the pair a chemist would
expect to confuse; measured on Batch 1 they are only the fourth closest pair
(centroid distance 4.42 class radii) while Acetaldehyde and Acetone are the
closest at 1.02, and the model separates all six perfectly at source. See
`docs/why-acetaldehyde.md`.

The geometry agrees. Every target batch's Acetaldehyde centroid lands nearest
Batch 1's **Ethanol** centroid, eight times of nine; see `docs/why-acetaldehyde.md`.

## The count table, re-checked on 2026-10-06: it matches every batch under a permutation

The claim above that Table 2 "matches the file in only one column" assumed the
table's columns are in label order. They are not. Matching the paper's sixty
cells (ten batches, six columns) against the file's per-label counts over all
720 column-to-label assignments:

| assignment | total absolute mismatch over 60 cells |
|---|---:|
| Ethanol=3, Ethylene=4, Ammonia=5, Acetaldehyde=2, Acetone=1, Toluene=6 | **9** |
| the adopted mapping (1=Ethanol, 2=Ammonia, 3=Ethylene, 4=Acetaldehyde, 5=Acetone, 6=Toluene) | 4371 |

The 9 is two typographical slips: Batch 5 Acetone printed 20 for 28, Batch 7
Ethylene printed 745 for 744. So the paper's Table 2 is the UCI page's own table
copied verbatim, and the data are the same files. What the table cannot settle
is the name-to-label mapping, because the UCI documentation contradicts itself:
its text gives one encoding and its count table implies another.

Which naming did the paper use for Fig. S3? Its dead class is "Acetaldehyde",
sent 100 % to "Ethanol". The file's dead class is label 4 (the 30-row class of
Batch 1), whose target centroid lands on label 1's. Under the text encoding
label 4 is Acetaldehyde and label 1 is Ethanol, which agrees with Fig. S3. Under
the count-table encoding label 4 is Ethylene and label 1 is Acetone, and the
paper's Ethylene at 0.95 would contradict the geometry every model here
measures. The paper therefore almost certainly named by the text encoding, and
the adopted mapping, which differs from it only in exchanging 2 and 3, keeps the
dead-class correspondence. Nothing numerical depends on this.

## What stays unexplained

The table's columns are not in label order (see the re-check above), so the
table and the dataset's stated encoding cannot both be read literally. The
encoding is the more direct statement about the files that are actually here,
and the paper's Fig. S3 is consistent with it; the table is kept as a fact about
counts, not about names.

`tests/test_baseline.py` therefore asserts the per-label **counts** - a fact
about the files - and the adopted mapping, rather than asserting a correspondence
between counts and names.
