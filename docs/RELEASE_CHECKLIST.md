# Release checklist

## Before the first push

1. **Fill the author block.** Done for this release: `.zenodo.json` (`creators`)
   and `CITATION.cff` (`authors`) carry Huang Zihao and Yan Chen with both
   affiliations, matching the manuscript. What is still missing is the
   corresponding author's e-mail in `12_latex/manuscript.tex` (marked TODO) and
   the ORCID fields if desired.
2. **Decide the repository name** and put it in `CITATION.cff`
   (`repository-code`). The name used throughout these files is
   `metric-floor-release`.
3. **Confirm the licence choice.** Code MIT, data CC BY 4.0 is the split assumed
   here; edit `LICENSE`, `LICENSE-DATA` and `.zenodo.json` together if it changes.
4. **Re-read `docs/KNOWN_DISCREPANCIES.md`** and decide whether to correct items
   1–3 in the manuscript before archiving.

## Push

```bash
cd 13_github
git add -A
git commit -m "Release v1.0.0: artefacts and analysis code"
git remote add origin git@github.com:HanHanWorks/metric-floor-release.git
git push -u origin main
```

The existing history already contains the 32 floor-free counterfactual runs;
this is intentional and consistent with Section 4.8 of the manuscript. The
working tree is about 10 MB (616 tracked files), of which 554 are run artefacts
(424 floored + 130 floor-free). No file exceeds 1 MB, so no Git LFS is needed.

Before pushing, confirm that the data-availability statement says the metered
and forecast series are confidential and cannot be released, and that no
de-identified sample is promised.

## Zenodo

1. Log in to Zenodo with GitHub, open *Settings → GitHub*, and flip the switch for
   this repository. Zenodo then archives every GitHub release automatically.
2. On GitHub, draft a release: tag `v1.0.0`, title "Code and data release v1.0.0",
   and paste the abstract from `.zenodo.json`.
3. Publish. Zenodo mints a DOI (concept DOI for all versions, version DOI for this
   one) and adds the badge to the repository.
4. Put the **concept DOI** in the manuscript's data-availability statement, so the
   statement stays valid across versions, and the version DOI in the reference list.
5. Update `.zenodo.json` (`related_identifiers`, `isSupplementTo`) with the article
   DOI once the paper is accepted, and `CITATION.cff` with both.

## After the DOI exists

- Add the DOI badge to the top of `README.md`.
- Replace `<this repository>` in the README quick start with the clone URL.
- If the paper is accepted with different table numbering, update
  `docs/FILE_MAP.md`; it is the only place that hard-codes table numbers.

## Verify the archive before you tag

```bash
git clone <the pushed repository> /tmp/check && cd /tmp/check
python -m pip install -r requirements.txt
python replication/verify_paper_numbers.py     # expect 142/142, no diff rows
make figures                                   # expect five figure pairs
```

A fresh clone is the only way to catch a file that was left untracked by
`.gitignore`.
