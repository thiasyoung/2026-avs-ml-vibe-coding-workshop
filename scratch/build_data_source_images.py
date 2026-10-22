"""Build NIST source cards from official images and an actual WebBook record.

The HTEM image is a separately captured public Finder screenshot. This script
does not execute notebooks, change datasets, or generate simulated spectra.
"""
from io import BytesIO
from pathlib import Path

from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
import requests

ASSETS = Path(__file__).resolve().parents[1] / "docs/data_sources"
FONT_DIR = Path(r"C:\Windows\Fonts")
BASE = "https://webbook.nist.gov"
BP_URL = BASE + "/cgi/cbook.cgi?ID=C64175&Units=SI&Mask=4"
NAVY, BLUE, GRAY = "#142d4e", "#086b9a", "#526174"


def response(url):
    result = requests.get(url, timeout=30)
    result.raise_for_status()
    return result


def picture(url):
    return Image.open(BytesIO(response(url).content)).convert("RGBA")


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ("arialbd.ttf" if bold else "arial.ttf")), size)


def paste_centered(canvas, image, box):
    image = image.copy()
    image.thumbnail((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
    canvas.paste(image, (box[0] + (box[2] - box[0] - image.width) // 2,
                         box[1] + (box[3] - box[1] - image.height) // 2), image)


def card(subtitle):
    canvas = Image.new("RGB", (1400, 760), "white")
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 1400, 185), fill=NAVY)
    paste_centered(canvas, picture(BASE + "/chemistry/img/touch-icon-192x192.png"), (32, 18, 186, 172))
    draw.text((215, 34), "NIST Chemistry WebBook", font=font(47, True), fill="white")
    draw.text((218, 101), "SRD 69  |  " + subtitle, font=font(28), fill="#d7ebfa")
    draw.text((46, 213), "Example source record: ethanol (CAS 64-17-5)", font=font(31, True), fill=NAVY)
    draw.line((46, 274, 1354, 274), fill="#d8e3ec", width=2)
    draw.text((46, 700), "webbook.nist.gov  |  Official source images and data; workshop layout", font=font(23), fill=GRAY)
    return canvas, draw


def build():
    soup = BeautifulSoup(response(BP_URL).content, "html.parser")
    table = soup.find("table", attrs={"aria-label": "One dimensional data"})
    boiling = None
    for row in table.find_all("tr"):
        values = [cell.get_text("", strip=True) for cell in row.find_all("td")]
        if values and values[0] == "Tboil":
            boiling = values
            break
    if boiling is None or boiling[2] != "K":
        raise ValueError("Expected an actual kelvin boiling-point entry from NIST.")

    canvas, draw = card("Thermophysical property records")
    structure = picture(BASE + "/cgi/cbook.cgi?Struct=C64175&Type=Color")
    structure = structure.resize((structure.width * 3, structure.height * 3), Image.Resampling.LANCZOS)
    paste_centered(canvas, structure, (70, 320, 440, 540))
    draw.text((147, 558), "C2H6O  |  CCO", font=font(30), fill=GRAY)
    draw.text((575, 322), "Normal boiling point", font=font(32, True), fill=NAVY)
    draw.text((575, 379), boiling[1] + " " + boiling[2], font=font(57, True), fill=BLUE)
    draw.text((575, 467), "NIST tabulated average (not a model prediction)", font=font(24), fill=GRAY)
    draw.text((575, 529), "Molecular identity + phase-change measurements", font=font(23), fill=NAVY)
    draw.text((575, 567), "are the source of the boiling-point workshop table.", font=font(23), fill=NAVY)
    canvas.save(ASSETS / "nist_webbook_boiling_point.png", optimize=True)

    canvas, draw = card("Infrared spectra and molecular identities")
    spectrum = picture(BASE + "/cgi/cbook.cgi?Spec=C64175&Index=0&Type=IR")
    # Preserve the official plot, including axes/units; do not synthesize a curve.
    paste_centered(canvas, spectrum.resize((636, 382), Image.Resampling.LANCZOS), (50, 292, 700, 682))
    draw.text((760, 345), "Native IR spectrum", font=font(34, True), fill=NAVY)
    draw.text((760, 415), "Keep coordinates, intensities", font=font(28), fill=GRAY)
    draw.text((760, 458), "and original units at acquisition.", font=font(28), fill=GRAY)
    draw.text((760, 536), "Conversion and resampling", font=font(26), fill=BLUE)
    draw.text((760, 576), "are notebook 01 decisions.", font=font(26), fill=BLUE)
    canvas.save(ASSETS / "nist_webbook_ir.png", optimize=True)
    print("Built both NIST source images from the live official record and plot.")


if __name__ == "__main__":
    build()
