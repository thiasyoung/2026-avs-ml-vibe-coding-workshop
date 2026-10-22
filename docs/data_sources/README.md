# Data-source images

Local illustrations for the three acquisition notebooks, assembled/captured on
October 21, 2026. They identify the repositories, not a fresh workshop acquisition.
Images display offline in VS Code and on GitHub without running a code cell.

| Image | Original source | Workshop use |
| --- | --- | --- |
| `nist_webbook_boiling_point.png` | [NIST Chemistry WebBook: ethanol, phase-change data](https://webbook.nist.gov/cgi/cbook.cgi?ID=C64175&Units=SI&Mask=4) | Boiling-point records and molecular identities |
| `nist_webbook_ir.png` | [NIST Chemistry WebBook: ethanol, infrared spectrum](https://webbook.nist.gov/cgi/cbook.cgi?ID=C64175&Units=SI&Type=IR-SPEC&Index=0) | Native IR spectra and molecular identities |
| `htem_database.png` | [High Throughput Experimental Materials database](https://htem.nlr.gov/) | Deposition recipes, substrate locations and scalar property measurements |

The two NIST images are **workshop source cards, not screenshots of the website**.
They combine the official [WebBook icon](https://webbook.nist.gov/chemistry/img/touch-icon-192x192.png)
with its [ethanol structure image](https://webbook.nist.gov/cgi/cbook.cgi?Struct=C64175&Type=Color),
tabulated normal boiling point, or [official spectrum plot](https://webbook.nist.gov/cgi/cbook.cgi?Spec=C64175&Index=0&Type=IR).
The plot is resized without altering its curve or axes; its transmittance remains
transmittance. The tabulated average is fetched from the actual source page, not
invented or taken from a model. This is illustrative, not a claim that the workshop
data use exactly that aggregate value or that every spectrum has the same units.

The HTEM image is a screenshot of its public Finder welcome screen. No private
account or authenticated page was used. Source links and credits appear in captions.
Original content/branding belong to their respective owners; no endorsement is
implied. NIST Standard Reference Data is not assumed to be public domain: see
the [WebBook copyright notice](https://webbook.nist.gov/chemistry/#Notes).

Rebuild the NIST cards on Windows with the workshop environment:
`uv run --locked python scratch/build_data_source_images.py`.
This only updates the two PNG illustrations; it does not execute the notebooks or
change their datasets. Re-capture the HTEM image from the public site if its UI changes.
