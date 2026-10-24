"""Create a style-only IEEE conference (A4) template from a reference .docx.

Keeps page setup, styles, numbering (I./A. headings, "Fig. N." captions, "TABLE N" heads, "[N]"
references) and the final two-column section; removes all body content, embedded images and
document metadata of the reference paper.

    python tools/make_ieee_template.py <reference.docx>
"""
import sys
from pathlib import Path

from docx import Document

OUT = Path(__file__).resolve().parents[1] / "reports" / "templates" / "ieee_a4_template.docx"


def main(src: str) -> None:
    doc = Document(src)
    body = doc.element.body
    for child in list(body.iterchildren()):
        if not child.tag.endswith("}sectPr"):  # keep only the final section properties
            body.remove(child)
    for rid, rel in list(doc.part.rels.items()):
        if rel.reltype.endswith("/image"):
            doc.part.drop_rel(rid)
    cp = doc.core_properties
    cp.author = ""
    cp.title = ""
    cp.subject = ""
    cp.last_modified_by = ""
    cp.comments = ""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
