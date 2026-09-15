# Initial dataset inspection

Inspection date: 2026-09-06 (UTC). This is a structural inspection, not a full dataset validation or PCA analysis.

## Discovery and format

The dataset is already extracted at `Dataset/`; no archive or dataset documentation was found in the project. The empty lowercase `dataset/` directory was left untouched. Ten files were discovered:

| Batch | File | Rows |
|---:|---|---:|
| 1 | `Dataset/batch1.dat` | 445 |
| 2 | `Dataset/batch2.dat` | 1,244 |
| 3 | `Dataset/batch3.dat` | 1,586 |
| 4 | `Dataset/batch4.dat` | 161 |
| 5 | `Dataset/batch5.dat` | 197 |
| 6 | `Dataset/batch6.dat` | 2,300 |
| 7 | `Dataset/batch7.dat` | 3,613 |
| 8 | `Dataset/batch8.dat` | 294 |
| 9 | `Dataset/batch9.dat` | 470 |
| 10 | `Dataset/batch10.dat` | 3,600 |

Files are plain-text, LIBSVM-style records. Each inspected record has one leading integer label token followed by 128 `index:value` feature tokens. Across all records, labels are 1–6, feature indices span 1–128, each index occurs in the dataset, and no malformed, out-of-order, or non-129-token rows were detected by the initial structural scan.

The record layout exposes no separate concentration field: after the single label token, all remaining tokens are indexed features. No project documentation or sidecar metadata explicitly providing concentration, device ID, humidity, or sample-level timestamps was found. This does not prove those concepts are absent from the dataset's external provenance; implementations must not infer them from the present files.

## Comparison with expectations

The scan found 13,910 rows, 10 batch files, labels 1–6, and 128 indexed features, matching the expected top-level dimensions. The 128 values are treated as 16 sensors × 8 extracted features, not as raw time samples. No immediate structural discrepancy was found, but semantic validation and numerical quality checks remain future work.

## Input identities (SHA-256)

```text
f346beee8e0c5e31ac5961845b6d96a70dc1ccf799481592fb2f0d96a81952e6  Dataset/batch1.dat
07f7e94a9bf4377240f9b230d20c750932fc785d4766b6a4b7fc370a035c825a  Dataset/batch2.dat
a4c7a1a6744df32f0ca139c75c3b4dade7cc99f57a7785d28cb0f62577dd2061  Dataset/batch3.dat
bd76f86be34ffe89f46a27c6fc2b5b8c6ebfcf984dff4b3e5befc76d98034b7f  Dataset/batch4.dat
3f95cb6e4a39a94bbacd2f1984f1754c9fb5eb3221812975ad70edbcea7abaec  Dataset/batch5.dat
83348c504105a5aa5264d1209f2a5d7ba7d9c8bcae60290395f201ac4708ff95  Dataset/batch6.dat
3168cb56d5c9bc29c36184e2e73c9f3474adb3c4a893b1ce0d88b6c47698cfdb  Dataset/batch7.dat
296346b932893ea18c513e23ac5b660f35e1c499fde8254d4804a2ce8a1b4ca7  Dataset/batch8.dat
e019f11f4fa8336ab1f41f7503eb456b3679da0a84eb8032fe6cef90f0547826  Dataset/batch9.dat
30011067c7c05c2b01038f84c73af0c6f1820b622fa2b0ad89111cb4c649f791  Dataset/batch10.dat
```
