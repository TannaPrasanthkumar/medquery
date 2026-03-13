# MediQuery — Data Setup Guide

## Step 1: Clone MedQuAD Dataset

```bash
cd data/
git clone https://github.com/abachaa/MedQuAD.git
```

This gives you 12 folders of XML files with 47,457 medical Q&A pairs from NIH.

## Step 2: Folder Structure After Clone

```
data/
└── MedQuAD/
    ├── 1_CancerGov_QA/
    ├── 2_GARD_QA/
    ├── 3_GHR_QA/
    ├── 4_MPlus_Health_Topics_QA/
    ├── 5_NIDDK_QA/
    ├── 6_NINDS_QA/
    ├── 7_SeniorHealth_QA/
    ├── 8_NHLBI_QA_XML/
    └── 9_CDC_QA/
```

## Step 3: In the App Sidebar

- Set the **MedQuAD folder path** to: `./data/MedQuAD`
- Click **"Build Knowledge Base"**
- Wait ~1-2 minutes for indexing (done once, then cached)

## Notes

- MedQuAD is licensed under **CC BY 4.0** (free to use with attribution)
- The app processes up to 60 XML files by default (configurable in `utils/rag_utils.py`)
- ChromaDB persists the index in `./chroma_db/` — no need to rebuild every run
- Folders 10, 11, 12 have answers removed (MedlinePlus copyright) — these are skipped automatically
