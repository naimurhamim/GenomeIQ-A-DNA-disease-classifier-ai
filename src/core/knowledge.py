"""Disease metadata + gene marker knowledge base used by the API."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DiseaseInfo:
    """Public-facing description of a disease class."""

    name: str
    icon: str
    color: str
    short_description: str
    long_description: str
    key_genes: tuple[str, ...]
    pathways: tuple[str, ...]
    references: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "icon": self.icon,
            "color": self.color,
            "short_description": self.short_description,
            "long_description": self.long_description,
            "key_genes": list(self.key_genes),
            "pathways": list(self.pathways),
            "references": list(self.references),
        }


DISEASE_INFO: dict[str, DiseaseInfo] = {
    "Cancer": DiseaseInfo(
        name="Cancer",
        icon="🔴",
        color="#ef4444",
        short_description="Markers of oncogenic gene expression detected.",
        long_description=(
            "Cancer is characterised by uncontrolled cell growth driven by mutations in "
            "tumour suppressor genes (e.g. TP53, BRCA1/2) and proto-oncogenes (e.g. EGFR, MYC). "
            "Detection is based on k-mer signatures associated with these gene families."
        ),
        key_genes=("BRCA1", "BRCA2", "TP53", "EGFR", "MYC", "KRAS", "PTEN"),
        pathways=("DNA damage response", "Cell cycle", "PI3K/AKT", "p53 signalling"),
        references=(
            "https://www.ncbi.nlm.nih.gov/gene/672",  # BRCA1
            "https://www.ncbi.nlm.nih.gov/gene/7157",  # TP53
        ),
    ),
    "Diabetes": DiseaseInfo(
        name="Diabetes",
        icon="🟠",
        color="#f59e0b",
        short_description="Markers of insulin / glucose regulation gene expression.",
        long_description=(
            "Diabetes mellitus involves dysregulation of insulin secretion and glucose "
            "homeostasis. Relevant genes include INS, INSR, GCK and HNF1A."
        ),
        key_genes=("INS", "INSR", "GCK", "HNF1A", "HNF4A", "PPARG"),
        pathways=("Insulin signalling", "Glucose metabolism", "Pancreatic beta-cell function"),
        references=(
            "https://www.ncbi.nlm.nih.gov/gene/3630",  # INS
            "https://www.ncbi.nlm.nih.gov/gene/2645",  # GCK
        ),
    ),
    "Alzheimers": DiseaseInfo(
        name="Alzheimers",
        icon="🟣",
        color="#a855f7",
        short_description="Markers associated with neurodegenerative pathways.",
        long_description=(
            "Alzheimer's disease is driven by amyloid-beta accumulation and tau pathology. "
            "Key genes include APP, PSEN1, PSEN2 and APOE."
        ),
        key_genes=("APP", "PSEN1", "PSEN2", "APOE", "MAPT", "TREM2"),
        pathways=(
            "Amyloid-beta processing",
            "Tau phosphorylation",
            "Neuronal apoptosis",
        ),
        references=(
            "https://www.ncbi.nlm.nih.gov/gene/351",  # APP
            "https://www.ncbi.nlm.nih.gov/gene/348",  # APOE
        ),
    ),
    "Normal": DiseaseInfo(
        name="Normal",
        icon="🟢",
        color="#10b981",
        short_description="No disease-associated markers detected.",
        long_description=(
            "Sequence patterns match housekeeping or non-pathogenic gene profiles "
            "(e.g. ACTB, GAPDH). This does not rule out rare or novel variants."
        ),
        key_genes=("ACTB", "GAPDH", "B2M", "HPRT1"),
        pathways=("Cytoskeleton", "Glycolysis", "Cellular housekeeping"),
        references=("https://www.ncbi.nlm.nih.gov/gene/60",),  # ACTB
    ),
}


def get_disease_info(name: str) -> DiseaseInfo | None:
    return DISEASE_INFO.get(name)


def list_diseases() -> list[DiseaseInfo]:
    return list(DISEASE_INFO.values())
