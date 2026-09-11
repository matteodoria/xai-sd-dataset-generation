"""Render a Markdown document to a print-ready HTML file.

Produces a standalone page styled for paper rather than for screen: serif body,
A4 margins, sections starting on a new page, and tables and code blocks kept from
splitting across pages. Open the result in a browser and use Print -> Save as PDF.

Usage:
    python Results/docs/render.py <file.md>
"""

import pathlib
import sys

import markdown

STYLE = """
@page {
    size: A4;
    margin: 2cm 2.2cm 2.4cm 2.2cm;
}

body {
    font-family: Georgia, "Iowan Old Style", "Times New Roman", serif;
    font-size: 11pt;
    line-height: 1.55;
    color: #1a1a1a;
    max-width: 42em;
    margin: 2em auto;
    padding: 0 1em;
    hyphens: auto;
}

h1, h2, h3, h4 {
    font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.25;
    color: #000;
}

h1 { font-size: 20pt; margin: 0 0 0.2em; }
h2 { font-size: 14pt; margin: 1.8em 0 0.6em; border-bottom: 1px solid #ddd;
     padding-bottom: 0.25em; }
h3 { font-size: 12pt; margin: 1.4em 0 0.4em; }

/* Each numbered section starts on a fresh page; the first one must not, or the
   opening page would hold nothing but the title. :first-of-type is used rather
   than a sibling selector because the intro paragraph and a rule sit in between,
   and any adjacency rule would break the moment that changes. */
h2 { page-break-before: always; }
h2:first-of-type { page-break-before: avoid; }

/* Never strand a heading at the foot of a page, and never split a table, a code
   block or a quotation across two. */
h1, h2, h3, h4 { page-break-after: avoid; }
pre, table, blockquote { page-break-inside: avoid; }
p { orphans: 3; widows: 3; }

code {
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 0.88em;
    background: #f4f4f4;
    padding: 0.1em 0.3em;
    border-radius: 3px;
}

pre {
    background: #f7f7f7;
    border-left: 3px solid #ccc;
    padding: 0.7em 1em;
    overflow-x: auto;
    font-size: 0.85em;
    line-height: 1.4;
}
pre code { background: none; padding: 0; font-size: 1em; }

table {
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
    font-size: 0.92em;
}
th, td { border: 1px solid #ccc; padding: 0.35em 0.6em; text-align: left; }
th { background: #f0f0f0; font-family: -apple-system, Arial, sans-serif;
     font-size: 0.92em; }

blockquote {
    margin: 1.2em 0;
    padding: 0.4em 0 0.4em 1.1em;
    border-left: 3px solid #999;
    color: #333;
    font-style: italic;
}

hr { border: none; border-top: 1px solid #ddd; margin: 2em 0; }

/* Links are useless on paper: keep them readable as plain text. */
a { color: #1a1a1a; text-decoration: none; border-bottom: 1px dotted #999; }

ul, ol { padding-left: 1.4em; }
li { margin: 0.3em 0; }

@media print {
    body { margin: 0; max-width: none; padding: 0; font-size: 10.5pt; }
    a { border-bottom: none; }
}
"""

TEMPLATE = """<!DOCTYPE html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{style}</style>
</head>
<body>
{body}
</body>
</html>
"""


def main():
    source = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "Results/docs/percorso.md")

    body = markdown.markdown(
        source.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "sane_lists", "smarty"])

    destination = source.with_suffix(".html")
    destination.write_text(
        TEMPLATE.format(title=source.stem, style=STYLE, body=body),
        encoding="utf-8")

    print(f"Written: {destination}")
    print("Open it in a browser and use Print -> Save as PDF.")
    print("In the print dialog, turn off headers and footers to drop the URL "
          "and date from every page.")


if __name__ == "__main__":
    main()
