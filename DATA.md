# Data

None of the data is redistributed here. Download each dataset from its source and respect its licence.

| Dataset | Country | Photos used | Patients | Access | Licence | Source | Cite |
|---|---|---|---|---|---|---|---|
| CODE (Version 5) | India | 5,486 | 694 | Open | see Figshare page | https://doi.org/10.6084/m9.figshare.30550889 | doi:10.1016/j.dib.2026.113270 |
| SMART-OM | India | 2,442 | 315 | Open | CC BY-NC-ND 4.0 | https://doi.org/10.6084/m9.figshare.31341790 | doi:10.1038/s41597-026-06954-5 |
| Sri Lanka (Peradeniya) | Sri Lanka | 2,988 | 712 | On request (Zenodo) | CC BY-NC-ND 4.0 | https://zenodo.org/records/10664056 | doi:10.1016/j.oraloncology.2024.106946 |
| Karnataka | India | 323 | – | Open | see source page | see paper reference | see paper |

Only unannotated photographs were used. Exact duplicates (MD5) and near-duplicates (difference hash) were removed.

## Split files (`splits/`)

- `india_karnataka_folds.csv`: one row per photograph, keyed by the image MD5 hash. `set` = `internal` (CODE + SMART-OM, 5 folds) or `karnataka` (external test only). `patient_group` keeps all photos of one person in one fold.
- `srilanka_folds.csv`: one row per photograph, keyed by the Sri Lankan image name and patient ID.

To reproduce the splits, compute the MD5 of each downloaded image and join on `image_md5`.
