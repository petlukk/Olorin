# Third-party assets

Olorin's runtime dependencies are only `libc` and `libloading`. The one
bundled third-party *asset* is a font, embedded in the web UI:

## DejaVu Sans Mono (subset)

`web/chat.html` embeds a small subset of **DejaVu Sans Mono** (ASCII + Box
Drawing + Block Elements glyphs, ~9 KB WOFF) as a base64 `@font-face`. It
renders the block-bar charts (`▁▂▃▄▅▆▇█`, `─`) at uniform monospace width on
every client OS, so the grid doesn't shear on machines that lack a
block-capable monospace font (e.g. a Windows browser viewing a remote Olorin).

- **Font:** DejaVu Sans Mono — https://dejavu-fonts.github.io/
- **License:** DejaVu Fonts License (based on the Bitstream Vera Fonts License
  and the Arev Fonts License) — permissive; the fonts and derivatives may be
  embedded and redistributed, including in commercial products, provided the
  copyright and license notices are retained. The fonts are not sold on their
  own.
- **Copyright:** Bitstream Vera fonts © 2003 Bitstream, Inc.; DejaVu changes
  are in the public domain. Arev fonts © 2006 Tavmjong Bah.
- Full license text: https://dejavu-fonts.github.io/License.html

Only the subsetted glyphs are embedded; the original font is not redistributed
in full.
