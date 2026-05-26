"""Curated genomic knowledge base for retrieval.

Each entry is a self-contained "fact" used as a retrieval document. Keep them
short and focused so the embedding model can match queries precisely. The
content is paraphrased from public sources (NCBI Gene, Ensembl, OMIM) and is
reproduced here in summary form rather than verbatim, well within
fair-use limits.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Fact:
    """A retrievable knowledge document."""

    fact_id: str
    topic: str  # e.g. disease name or gene symbol
    category: str  # "disease", "gene", "pathway", "general"
    title: str
    content: str
    source: str | None = None


# ---------------------------------------------------------------------------
# Disease overviews
# ---------------------------------------------------------------------------

DISEASE_FACTS: list[Fact] = [
    Fact(
        fact_id="disease.cancer.overview",
        topic="Cancer",
        category="disease",
        title="Cancer overview",
        content=(
            "Cancer is a group of diseases characterised by uncontrolled cell growth and "
            "the ability of malignant cells to invade other tissues. At the molecular "
            "level it is driven by acquired mutations in tumour-suppressor genes "
            "(such as TP53, BRCA1, BRCA2, PTEN) and proto-oncogenes (such as EGFR, KRAS, "
            "MYC). The classifier flags sequences whose k-mer signatures resemble these "
            "gene families."
        ),
        source="NCBI / NCI overview",
    ),
    Fact(
        fact_id="disease.diabetes.overview",
        topic="Diabetes",
        category="disease",
        title="Diabetes mellitus overview",
        content=(
            "Diabetes mellitus is a group of metabolic diseases marked by chronic "
            "hyperglycaemia due to defects in insulin secretion, insulin action, or "
            "both. Monogenic forms (MODY) involve genes such as HNF1A, HNF4A, and GCK. "
            "Type 1 disease has strong autoimmune and HLA components. Sequences "
            "matching INS, INSR, GCK or PPARG patterns are flagged in this class."
        ),
        source="NCBI / OMIM",
    ),
    Fact(
        fact_id="disease.alzheimers.overview",
        topic="Alzheimers",
        category="disease",
        title="Alzheimer's disease overview",
        content=(
            "Alzheimer's disease is a progressive neurodegenerative disorder. "
            "Pathologically it features extracellular amyloid-beta plaques and "
            "intracellular hyperphosphorylated tau tangles. Familial early-onset cases "
            "involve mutations in APP, PSEN1, PSEN2; APOE-epsilon4 increases risk for "
            "late-onset disease. Sequences resembling these gene regions are flagged "
            "in this class."
        ),
        source="NCBI / OMIM",
    ),
    Fact(
        fact_id="disease.normal.overview",
        topic="Normal",
        category="disease",
        title="Normal class overview",
        content=(
            "The 'Normal' class collects sequences from housekeeping or constitutively "
            "expressed genes (such as ACTB, GAPDH, B2M, HPRT1). It serves as a baseline "
            "comparison against the disease-associated classes. A 'Normal' prediction "
            "does not rule out rare or novel pathogenic variants."
        ),
        source="MGI / housekeeping gene catalogues",
    ),
]


# ---------------------------------------------------------------------------
# Gene-level facts
# ---------------------------------------------------------------------------

GENE_FACTS: list[Fact] = [
    # Cancer genes
    Fact(
        fact_id="gene.brca1",
        topic="BRCA1",
        category="gene",
        title="BRCA1 — breast cancer 1, early onset",
        content=(
            "BRCA1 is a tumour-suppressor gene on chromosome 17q21. It encodes a protein "
            "involved in DNA double-strand break repair via homologous recombination. "
            "Loss-of-function mutations dramatically increase lifetime risk of breast "
            "and ovarian cancer."
        ),
        source="NCBI Gene 672",
    ),
    Fact(
        fact_id="gene.brca2",
        topic="BRCA2",
        category="gene",
        title="BRCA2 — breast cancer 2, early onset",
        content=(
            "BRCA2 cooperates with BRCA1 and RAD51 in homologous recombination repair. "
            "Pathogenic variants confer high risk of breast, ovarian, prostate and "
            "pancreatic cancers."
        ),
        source="NCBI Gene 675",
    ),
    Fact(
        fact_id="gene.tp53",
        topic="TP53",
        category="gene",
        title="TP53 — the guardian of the genome",
        content=(
            "TP53 encodes the p53 transcription factor. It coordinates cell-cycle "
            "arrest, DNA repair and apoptosis in response to genotoxic stress. TP53 "
            "is the most frequently mutated gene in human cancers; germline mutations "
            "cause Li-Fraumeni syndrome."
        ),
        source="NCBI Gene 7157",
    ),
    Fact(
        fact_id="gene.egfr",
        topic="EGFR",
        category="gene",
        title="EGFR — epidermal growth factor receptor",
        content=(
            "EGFR is a receptor tyrosine kinase. Activating mutations and amplification "
            "drive non-small-cell lung cancer and other malignancies, and are the target "
            "of inhibitors such as erlotinib, gefitinib and osimertinib."
        ),
        source="NCBI Gene 1956",
    ),
    Fact(
        fact_id="gene.kras",
        topic="KRAS",
        category="gene",
        title="KRAS — proto-oncogene",
        content=(
            "KRAS is a small GTPase in the RAS family. Mutations at G12, G13 and Q61 "
            "lock it in an active state, driving pancreatic, colorectal and lung "
            "cancers. KRAS-G12C inhibitors (e.g. sotorasib) are clinically approved."
        ),
        source="NCBI Gene 3845",
    ),
    Fact(
        fact_id="gene.myc",
        topic="MYC",
        category="gene",
        title="MYC — proto-oncogene transcription factor",
        content=(
            "MYC is a basic helix-loop-helix transcription factor regulating cell "
            "growth, metabolism and proliferation. Translocations (e.g. in Burkitt "
            "lymphoma) and amplification deregulate MYC across many cancer types."
        ),
        source="NCBI Gene 4609",
    ),
    Fact(
        fact_id="gene.pten",
        topic="PTEN",
        category="gene",
        title="PTEN — phosphatase and tensin homolog",
        content=(
            "PTEN is a tumour-suppressor lipid phosphatase that dephosphorylates "
            "PIP3, antagonising PI3K/AKT signalling. Loss promotes proliferation and "
            "survival; germline mutations cause Cowden syndrome."
        ),
        source="NCBI Gene 5728",
    ),
    # Diabetes genes
    Fact(
        fact_id="gene.ins",
        topic="INS",
        category="gene",
        title="INS — insulin",
        content=(
            "INS encodes preproinsulin. Mature insulin regulates glucose homeostasis "
            "by stimulating glucose uptake and storage. Mutations cause neonatal "
            "diabetes and rare maturity-onset diabetes of the young (MODY10)."
        ),
        source="NCBI Gene 3630",
    ),
    Fact(
        fact_id="gene.insr",
        topic="INSR",
        category="gene",
        title="INSR — insulin receptor",
        content=(
            "INSR is the cell-surface tyrosine-kinase receptor for insulin. "
            "Loss-of-function causes severe insulin resistance syndromes such as "
            "Donohue and Rabson-Mendenhall syndromes."
        ),
        source="NCBI Gene 3643",
    ),
    Fact(
        fact_id="gene.gck",
        topic="GCK",
        category="gene",
        title="GCK — glucokinase",
        content=(
            "GCK acts as the pancreatic beta-cell glucose sensor and the hepatic "
            "first step of glycolysis. Heterozygous loss-of-function causes MODY2 "
            "with stable mild fasting hyperglycaemia."
        ),
        source="NCBI Gene 2645",
    ),
    Fact(
        fact_id="gene.hnf1a",
        topic="HNF1A",
        category="gene",
        title="HNF1A — hepatocyte nuclear factor 1 alpha",
        content=(
            "HNF1A is a transcription factor critical for pancreatic beta-cell "
            "function. Heterozygous mutations cause MODY3 with progressive "
            "insulin secretory deficit, often responsive to sulphonylureas."
        ),
        source="NCBI Gene 6927",
    ),
    Fact(
        fact_id="gene.pparg",
        topic="PPARG",
        category="gene",
        title="PPARG — peroxisome proliferator activated receptor gamma",
        content=(
            "PPARG is a master regulator of adipogenesis and insulin sensitivity. "
            "Loss-of-function variants cause familial partial lipodystrophy with "
            "insulin resistance; thiazolidinediones are PPARG agonists used in "
            "type 2 diabetes."
        ),
        source="NCBI Gene 5468",
    ),
    # Alzheimer's genes
    Fact(
        fact_id="gene.app",
        topic="APP",
        category="gene",
        title="APP — amyloid precursor protein",
        content=(
            "APP is a transmembrane glycoprotein cleaved sequentially by beta- and "
            "gamma-secretases to generate amyloid-beta peptides that accumulate as "
            "plaques in Alzheimer's disease. Familial APP mutations cause "
            "early-onset autosomal dominant Alzheimer's."
        ),
        source="NCBI Gene 351",
    ),
    Fact(
        fact_id="gene.psen1",
        topic="PSEN1",
        category="gene",
        title="PSEN1 — presenilin 1",
        content=(
            "PSEN1 is the catalytic subunit of gamma-secretase. Mutations alter "
            "amyloid-beta cleavage ratios, increasing amyloidogenic Abeta42 production "
            "and causing the most common form of early-onset familial Alzheimer's."
        ),
        source="NCBI Gene 5663",
    ),
    Fact(
        fact_id="gene.psen2",
        topic="PSEN2",
        category="gene",
        title="PSEN2 — presenilin 2",
        content=(
            "PSEN2 is a homologue of PSEN1 contributing to gamma-secretase activity. "
            "Pathogenic variants are a less common cause of early-onset familial "
            "Alzheimer's disease."
        ),
        source="NCBI Gene 5664",
    ),
    Fact(
        fact_id="gene.apoe",
        topic="APOE",
        category="gene",
        title="APOE — apolipoprotein E",
        content=(
            "APOE encodes a lipid-transport protein expressed in the brain. The "
            "epsilon-4 allele is the strongest known genetic risk factor for "
            "late-onset Alzheimer's disease, while epsilon-2 is protective."
        ),
        source="NCBI Gene 348",
    ),
    Fact(
        fact_id="gene.mapt",
        topic="MAPT",
        category="gene",
        title="MAPT — microtubule-associated protein tau",
        content=(
            "MAPT encodes tau, which stabilises neuronal microtubules. "
            "Hyperphosphorylated tau forms the neurofibrillary tangles seen in "
            "Alzheimer's disease and frontotemporal dementia."
        ),
        source="NCBI Gene 4137",
    ),
    Fact(
        fact_id="gene.trem2",
        topic="TREM2",
        category="gene",
        title="TREM2 — triggering receptor expressed on myeloid cells 2",
        content=(
            "TREM2 is a microglial receptor regulating phagocytosis and "
            "neuroinflammation. Rare variants (e.g. R47H) substantially increase "
            "Alzheimer's disease risk."
        ),
        source="NCBI Gene 54209",
    ),
    # Normal / housekeeping genes
    Fact(
        fact_id="gene.actb",
        topic="ACTB",
        category="gene",
        title="ACTB — beta-actin",
        content=(
            "ACTB encodes beta-actin, a ubiquitously expressed cytoskeletal protein. "
            "It is a canonical housekeeping gene routinely used as a reference in "
            "expression studies."
        ),
        source="NCBI Gene 60",
    ),
    Fact(
        fact_id="gene.gapdh",
        topic="GAPDH",
        category="gene",
        title="GAPDH — glyceraldehyde-3-phosphate dehydrogenase",
        content=(
            "GAPDH catalyses a key step in glycolysis. It is a constitutively "
            "expressed housekeeping gene used as a normalisation control in "
            "molecular assays."
        ),
        source="NCBI Gene 2597",
    ),
]


# ---------------------------------------------------------------------------
# Methodology / project facts (so the chat can answer questions about itself)
# ---------------------------------------------------------------------------

METHOD_FACTS: list[Fact] = [
    Fact(
        fact_id="method.tfidf",
        topic="GenomeIQ method",
        category="general",
        title="TF-IDF k-mer classifier",
        content=(
            "GenomeIQ's primary classifier uses TF-IDF on 4-grams of overlapping "
            "6-mer DNA tokens. Predictions come from a soft-voting ensemble of a "
            "Random Forest, a calibrated Linear SVM and a Logistic Regression head. "
            "5-fold group-aware cross-validation reports macro F1 of 0.96 ± 0.02."
        ),
        source="GenomeIQ project",
    ),
    Fact(
        fact_id="method.dnabert2",
        topic="GenomeIQ method",
        category="general",
        title="DNABERT-2 transformer backbone",
        content=(
            "GenomeIQ also supports DNABERT-2, a 117 M-parameter genomic transformer "
            "(MosaicBERT architecture, BPE tokenizer). It can be used as a frozen "
            "feature extractor with a logistic-regression head, or end-to-end "
            "fine-tuned for the four-class disease classification task."
        ),
        source="GenomeIQ project",
    ),
    Fact(
        fact_id="method.ood",
        topic="GenomeIQ method",
        category="general",
        title="Out-of-distribution detection",
        content=(
            "GenomeIQ flags sequences whose maximum cosine similarity to the "
            "training corpus falls below the 5th-percentile threshold. Such "
            "sequences are reported as 'high risk / unreliable' so that the user "
            "can interpret confidence accordingly."
        ),
        source="GenomeIQ project",
    ),
    Fact(
        fact_id="method.augmentation",
        topic="GenomeIQ method",
        category="general",
        title="Data augmentation pipeline",
        content=(
            "Training sequences are augmented with overlapping sliding windows of "
            "300, 600 and 1200 bases plus their reverse complements. Near-duplicates "
            "are removed with a MinHash-style Jaccard signature. Splits are "
            "constructed group-aware (parent-sequence level) to avoid leakage."
        ),
        source="GenomeIQ project",
    ),
    Fact(
        fact_id="method.disclaimer",
        topic="GenomeIQ disclaimer",
        category="general",
        title="Important disclaimer",
        content=(
            "GenomeIQ is a research preview only. Predictions, similarity scores "
            "and mutation analyses are derived from a finite training corpus and "
            "are not intended for clinical decision-making."
        ),
        source="GenomeIQ project",
    ),
]


def all_facts() -> list[Fact]:
    """Return the full curated knowledge base."""
    return DISEASE_FACTS + GENE_FACTS + METHOD_FACTS
