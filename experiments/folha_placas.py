"""Folha de placas SIMULADAS para imprimir e mostrar à webcam (teste em ambiente controlado).

Gera, em experiments/folha_placas/:
- folha_1.png, folha_2.png...: A4 em 300 dpi, dividida na horizontal, uma placa em cada metade
  (folha_1 = frente, folha_2 = verso). Imprimir em escala 100% / "tamanho real".
- gabarito.csv: id, texto, tipo e arquivo de cada placa (lido por experiments/webcam_placas.py).

As placas são desenhadas (sem imagem de dataset nem de carro real), com texto aleatório no formato
válido. Em cada folha: em cima Mercosul (branca, faixa azul "BRASIL"), embaixo modelo antigo (cinza,
"CIDADE - UF"). Tamanho: 40 x 13 cm reais impressos a 47,5% (19 x 6,2 cm).
A fonte não é a oficial (FE-Schrift / Mandatory): usa a Bahnschrift do Windows (estilo DIN).

Uso: .venv\\Scripts\\python.exe -m experiments.folha_placas [--folhas 2] [--seed 7]
"""

import argparse
import csv
import random
import string
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SAIDA = Path("experiments/folha_placas")
FONTE = r"C:\Windows\Fonts\bahnschrift.ttf"
DPI = 300
MM = DPI / 25.4  # pixels por milímetro
A4 = (round(210 * MM), round(297 * MM))
ESCALA = 0.475  # placa real 400 x 130 mm -> 190 x 61,75 mm no papel
AZUL, PRETO, BRANCO, CINZA, LINHA = (0, 51, 153), (17, 17, 17), (255, 255, 255), (200, 200, 200), (190, 190, 190)


def mm(valor_real_mm: float) -> int:
    """Medida da placa REAL (mm) -> pixels no papel, já com a escala."""
    return round(valor_real_mm * ESCALA * MM)


def texto_aleatorio(rng: random.Random, mercosul: bool) -> str:
    letras = "".join(rng.choices(string.ascii_uppercase, k=3))
    if mercosul:
        return f"{letras}{rng.randint(0, 9)}{rng.choice(string.ascii_uppercase)}{rng.randint(0, 99):02d}"
    return f"{letras}{rng.randint(0, 9999):04d}"


def fonte(variacao: str, tamanho: int) -> ImageFont.FreeTypeFont:
    f = ImageFont.truetype(FONTE, tamanho)
    f.set_variation_by_name(variacao)
    return f


def caber(texto: str, variacao: str, largura: int, altura: int) -> ImageFont.FreeTypeFont:
    """Maior fonte em que o texto cabe na caixa largura x altura (altura = da letra maiúscula)."""
    tamanho = altura * 2
    while tamanho > 10:
        f = fonte(variacao, tamanho)
        x0, y0, x1, y1 = f.getbbox(texto)
        if x1 - x0 <= largura and y1 - y0 <= altura:
            return f
        tamanho -= 4
    return fonte(variacao, 10)


def centralizar(d: ImageDraw.ImageDraw, texto: str, f, cx: int, cy: int, cor) -> None:
    x0, y0, x1, y1 = d.textbbox((0, 0), texto, font=f)
    d.text((cx - (x0 + x1) / 2, cy - (y0 + y1) / 2), texto, font=f, fill=cor)


def desenhar_placa(texto: str, mercosul: bool) -> Image.Image:
    """Placa de carro: 400 x 130 mm reais; caracteres de ~63 mm de altura."""
    w, h = mm(400), mm(130)
    img = Image.new("RGB", (w, h), BRANCO)
    d = ImageDraw.Draw(img)
    fundo = BRANCO if mercosul else CINZA
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=mm(10), fill=fundo, outline=PRETO, width=mm(4))
    if mercosul:
        faixa = mm(30)
        borda = mm(4)
        d.rounded_rectangle((borda, borda, w - borda, faixa), radius=mm(6), fill=AZUL)
        d.rectangle((borda, faixa - mm(8), w - borda, faixa), fill=AZUL)
        centralizar(d, "BRASIL", caber("BRASIL", "Bold", w, mm(14)), w // 2, (borda + faixa) // 2, BRANCO)
        corpo_topo, legenda = faixa, texto
    else:
        corpo_topo = mm(28)
        centralizar(d, "CIDADE - UF", caber("CIDADE - UF", "SemiBold", w, mm(11)), w // 2, mm(16), PRETO)
        legenda = f"{texto[:3]}-{texto[3:]}"
    f = caber(legenda, "Bold Condensed" if mercosul else "SemiBold Condensed", int(w * 0.9), mm(63))
    centralizar(d, legenda, f, w // 2, (corpo_topo + h) // 2 + mm(2), PRETO)
    return img


def folha(placas: list[tuple[int, str, bool]]) -> Image.Image:
    """A4 dividido na horizontal: uma placa centralizada em cada metade, linha tracejada no meio."""
    pagina = Image.new("RGB", A4, BRANCO)
    d = ImageDraw.Draw(pagina)
    meia = A4[1] // 2
    for x in range(0, A4[0], 40):  # linha tracejada para dobrar/cortar
        d.line((x, meia, x + 20, meia), fill=LINHA, width=3)
    # sem rótulo de texto na página: o detector achava placas falsas nele (o id está no gabarito)
    for i, (_, texto, mercosul) in enumerate(placas):
        placa = desenhar_placa(texto, mercosul)
        cy = meia * i + meia // 2
        pagina.paste(placa, ((A4[0] - placa.width) // 2, cy - placa.height // 2))
    return pagina


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folhas", type=int, default=2, help="páginas A4 (2 = frente e verso)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    SAIDA.mkdir(parents=True, exist_ok=True)
    for antigo in SAIDA.glob("folha*"):  # versões anteriores (inclusive a folha.html com o RodoSol)
        antigo.unlink()

    linhas, paginas = [], []
    for n in range(1, args.folhas + 1):
        placas = []
        for mercosul in (True, False):
            pid = len(linhas) + 1
            texto = texto_aleatorio(rng, mercosul)
            placas.append((pid, texto, mercosul))
            linhas.append([pid, texto, "carro mercosur" if mercosul else "carro brazilian", f"folha_{n}.png"])
        paginas.append(folha(placas))
        paginas[-1].save(SAIDA / f"folha_{n}.png", dpi=(DPI, DPI))
    # PDF A4 para imprimir (uma página por folha, em ordem: frente, verso...)
    paginas[0].save(SAIDA / "folha_placas.pdf", save_all=True, append_images=paginas[1:], resolution=DPI)

    with open(SAIDA / "gabarito.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "texto", "tipo", "arquivo"])
        w.writerows(linhas)

    for n in range(1, args.folhas + 1):
        print((SAIDA / f"folha_{n}.png").resolve())
    print((SAIDA / "gabarito.csv").resolve())


if __name__ == "__main__":
    main()
