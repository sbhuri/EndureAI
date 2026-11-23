"""Reproduce the selected public FEMA extracts using two downloaded PDFs and Poppler."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    (
        "cpg101", "Developing and Maintaining Emergency Operations Plans (CPG 101 v3)",
        "2021-09", "https://www.fema.gov/sites/default/files/documents/fema_cpg-101-v3-developing-maintaining-eops.pdf",
        "https://www.govinfo.gov/content/pkg/GOVPUB-HS5_100-PURL-gpo173010/pdf/GOVPUB-HS5_100-PURL-gpo173010.pdf",
        (
            (60, 60, "Planning assumptions", "all-hazards", ["capacity"]),
            (61, 63, "Develop and Analyze Courses of Action", "all-hazards", ["capacity"]),
            (64, 65, "Identify Resources and Information Needs", "all-hazards", ["capacity"]),
        ),
    ),
    (
        "evacuation", "Planning Considerations: Evacuation and Shelter-in-Place",
        "2019-07", "https://www.fema.gov/sites/default/files/2020-07/planning-considerations-evacuation-and-shelter-in-place.pdf",
        "https://www.govinfo.gov/content/pkg/GOVPUB-HS-PURL-gpo146636/pdf/GOVPUB-HS-PURL-gpo146636.pdf",
        (
            (13, 13, "Evacuation Facilities", "all-hazards", ["capacity"]),
            (14, 15, "Critical Considerations: Access and Functional Needs", "all-hazards", ["access"]),
            (23, 23, "In Practice: Virginia Know Your Zone", "hurricane", ["evacuation"]),
            (26, 28, "Planning Principles", "all-hazards", ["evacuation", "access"]),
        ),
    ),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpg101", type=Path)
    parser.add_argument("evacuation", type=Path)
    args = parser.parse_args()
    sources = []
    for key, title, published, source, url, sections in DOCUMENTS:
        pdf = getattr(args, key)
        pages = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True).split("\f")
        sources.append({
            "id": key, "title": title, "published": published, "source_url": source,
            "retrieved_url": url, "pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
            "sections": [
                {"title": name, "hazard": hazard, "topics": topics,
                 "pages": [{"page": page, "text": pages[page - 1].strip()} for page in range(first, last + 1)]}
                for first, last, name, hazard, topics in sections
            ],
        })
    output = ROOT / "data/fema_guidance.json"
    output.write_text(json.dumps(sources, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
