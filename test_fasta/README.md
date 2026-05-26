# 🧪 Test FASTA Files

This folder contains pre-made FASTA files for testing the **Batch Upload** feature of GenomeIQ.

---

## 📁 Files

| File | Records | Purpose |
|------|---------|---------|
| `01_all_classes.fasta` | 4 | One sequence per disease class (Cancer, Diabetes, Alzheimers, Normal). All should predict correctly. |
| `02_cancer_variants.fasta` | 3 | Multiple Cancer-related sequences (BRCA1 variants + TP53). All should predict Cancer. |
| `03_mixed_valid_invalid.fasta` | 5 | Mix of valid sequences, invalid characters, too-short, and low-complexity. Tests error handling. Expected: 2 successful, 3 failed. |
| `04_diabetes_alzheimers.fasta` | 4 | Two Diabetes (INS, GCK) + two Alzheimers (APP, PSEN1) sequences. |
| `05_single_sequence.fasta` | 1 | A single BRCA1 sequence. Tests that single-record FASTA works. |

---

## ✅ Expected Results

### 01_all_classes.fasta
| ID | Expected class | Confidence |
|----|---------------|------------|
| seq_brca1 | Cancer | ~79% |
| seq_ins | Diabetes | ~96% |
| seq_app | Alzheimers | ~44% |
| seq_actb | Normal | ~79% |

### 02_cancer_variants.fasta
| ID | Expected class |
|----|---------------|
| brca1_variant1 | Cancer |
| brca1_variant2 | Cancer |
| tp53_fragment | Cancer |

### 03_mixed_valid_invalid.fasta
| ID | Expected |
|----|----------|
| valid_sequence_1 | Cancer (OK) |
| invalid_sequence | FAIL — no valid DNA bases |
| too_short | FAIL — too short |
| valid_sequence_2 | Normal (OK) |
| low_complexity | OK but with very low confidence |

### 04_diabetes_alzheimers.fasta
| ID | Expected class |
|----|---------------|
| ins_human | Diabetes |
| gck_fragment | Diabetes |
| app_human | Alzheimers |
| psen1_fragment | Alzheimers |

### 05_single_sequence.fasta
| ID | Expected class |
|----|---------------|
| single_brca1_test | Cancer |

---

## 🪜 How to Use

1. Open GenomeIQ at `http://127.0.0.1:8000`
2. Click **📦 Batch Upload** in the sidebar
3. Click the file picker
4. Select any `.fasta` file from this folder
5. Check the results table against the expected values above

---

## 🛠️ Creating Your Own Test Files

FASTA format:
```
>record_id optional description
ATGCATGCATGC...
ATGCATGCATGC...
>next_record_id
ATGCATGC...
```

Rules:
- Each record starts with `>` followed by an ID
- Sequence can span multiple lines
- Only A, T, G, C, N characters are valid (others are stripped)
- Minimum 20 bases per sequence
- Maximum 200,000 bases per sequence
