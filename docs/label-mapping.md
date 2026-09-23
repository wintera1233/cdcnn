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

## What stays unexplained

Under the adopted mapping the per-batch count table matches the file in only one
column, Toluene. Either the table's columns are not in label order, or it
describes a differently ordered release of the files. This is recorded rather
than resolved: no reading of the table can simultaneously satisfy the dataset's
own stated encoding, and the encoding is the more direct statement about the
files that are actually here.

`tests/test_baseline.py` therefore asserts the per-label **counts** - a fact
about the files - and the adopted mapping, rather than asserting a correspondence
between counts and names.
