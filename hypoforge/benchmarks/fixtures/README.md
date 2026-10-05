# PDF font check

`cff_font_check.pdf` is an original 1-page, 1,153-byte synthetic fixture generated
with fontTools 4.66.1 (`FontBuilder`, `T2CharStringPen`) and pypdf 6.19.0.

It embeds an empty-outline CFF Type1 font with glyphs `.notdef` and `B`.
Its custom CFF encoding maps character code 65 to glyph `B`, while the PDF font
dictionary declares WinAnsiEncoding and has no ToUnicode map. The content stream
is `BT /F1 12 Tf 72 720 Td (A) Tj ET`.

Correct text extraction returns `B`. Without CFF encoding support it returns
`A` and emits the missing-fontTools warning. This checks actual font decoding,
not only whether a package can be imported. It contains no third-party fonts,
paper content, credentials, or network references.
