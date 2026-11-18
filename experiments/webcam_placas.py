"""Teste manual com a webcam USB: detector + OCR do modo `fpo`, ao vivo, com janela e log.

Mostra na janela cada placa detectada, com o texto lido, as confianças e a largura em pixels.
Cores: VERDE = leitura igual a uma placa do gabarito; VERMELHO = passou nos limiares, mas o texto
não está no gabarito; CINZA = abaixo dos limiares (o serviço descartaria).
No topo: resolução, ms por quadro e a faixa de largura da placa parecida com a do RodoSol.

Uso:
  .venv\\Scripts\\python.exe -m experiments.webcam_placas            (webcam 0)
  .venv\\Scripts\\python.exe -m experiments.webcam_placas 1 --dshow --largura 1280 --altura 720
  ... --fpo-model models/ocr_fpo_br.onnx                             (OCR ajustado, se estiver aqui)
Teclas: q/Esc (ou fechar a janela no X) sai, s salva o quadro anotado.

Saídas em experiments/webcam/ (no .gitignore): leituras_<data>.csv e os quadros salvos.
Mostra a placa completa de propósito (ferramenta de desenvolvimento; use placas impressas).
"""

import argparse
import csv
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2

from app.camera.source import CameraSource
from app.config import get_settings
from app.recognition.fpo_reader import FastPlateReader
from app.recognition.plate_format import normalize_plate
from app.recognition.preprocessing import crop_region
from app.recognition.tf_models import DETECTOR_INPUT_WIDTH, TfPlateDetector

SAIDA = Path("experiments/webcam")
GABARITO = Path("experiments/folha_placas/gabarito.csv")
# No RodoSol (1280 px de largura) a placa de carro tem ~125 px; o detector reduz o quadro para
# 384 px (placa ~38 px) e foi treinado com zoom de até 2x. Largura parecida na entrada: ~35-80 px.
FAIXA_NA_ENTRADA = (35, 80)

VERDE, VERMELHO, CINZA, BRANCO = (0, 200, 0), (0, 0, 230), (160, 160, 160), (255, 255, 255)


def abrir_captura(dshow: bool, largura: int | None, altura: int | None):
    """capture_factory do CameraSource: aplica DirectShow e a resolução pedida."""

    def fabrica(fonte):
        cap = cv2.VideoCapture(fonte, cv2.CAP_DSHOW) if dshow else cv2.VideoCapture(fonte)
        if largura:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, largura)
        if altura:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, altura)
        return cap

    return fabrica


def ler_gabarito(caminho: Path) -> dict[str, str]:
    if not caminho.is_file():
        return {}
    with open(caminho, encoding="utf-8") as f:
        return {linha["texto"]: f"#{linha['id']}" for linha in csv.DictReader(f)}


def escrever(img, texto, pos, cor, escala=0.6):
    """Texto com contorno preto, legível sobre qualquer fundo."""
    cv2.putText(img, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 4)
    cv2.putText(img, texto, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, 1)


def main() -> None:
    cfg = get_settings()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("camera", nargs="?", default="0", help="índice da webcam (padrão 0)")
    ap.add_argument("--dshow", action="store_true", help="usar DirectShow (se a câmera não abrir)")
    ap.add_argument("--largura", type=int, help="largura pedida à câmera (ex.: 1280)")
    ap.add_argument("--altura", type=int, help="altura pedida à câmera (ex.: 720)")
    ap.add_argument("--fpo-model", default=cfg.fpo_model, help=f"OCR (padrão do .env: {cfg.fpo_model})")
    ap.add_argument("--gabarito", type=Path, default=GABARITO)
    args = ap.parse_args()

    min_det = cfg.min_confidence
    min_ocr = cfg.min_confidence if cfg.ocr_min_confidence is None else cfg.ocr_min_confidence
    gabarito = ler_gabarito(args.gabarito)
    print(f"Gabarito: {len(gabarito)} placas ({args.gabarito})" if gabarito else "Sem gabarito: só mostra as leituras")

    print("Carregando modelos (até ~1 min na 1ª vez)...")
    detector = TfPlateDetector.from_file(Path(cfg.detector_model_path), threshold=cfg.detector_threshold)
    reader = FastPlateReader.from_model(args.fpo_model)
    print(f"Detector: {cfg.detector_model_path} | OCR: {args.fpo_model} | limiares det {min_det} ocr {min_ocr}")

    SAIDA.mkdir(parents=True, exist_ok=True)
    carimbo = datetime.now().strftime("%Y%m%d_%H%M%S")
    arq_csv = SAIDA / f"leituras_{carimbo}.csv"
    colunas = ["instante", "quadro", "texto", "formato", "det", "ocr", "larg_px", "alt_px",
               "passou_limiares", "gabarito"]

    # sample_interval_s=0 -> passo 1: analisa todo quadro que conseguir (a inferência já limita o ritmo)
    fonte = CameraSource(args.camera, sample_interval_s=0, capture_factory=abrir_captura(args.dshow, args.largura, args.altura))
    quadros = salvos = 0
    aceitas: Counter[str] = Counter()  # leituras que o serviço aceitaria, por texto
    janela = "webcam_placas (q sai, s salva)"

    with open(arq_csv, "w", newline="", encoding="utf-8") as f:
        log = csv.writer(f)
        log.writerow(colunas)
        try:
            for frame in fonte.frames():
                inicio = time.perf_counter()
                quadros += 1
                alt, larg = frame.shape[:2]
                vista = frame.copy()

                for det in sorted(detector.detect(frame), key=lambda d: d.confidence, reverse=True):
                    b = det.bbox
                    recorte = crop_region(frame, b)
                    leitura = reader.read(recorte) if recorte is not None else None
                    texto = leitura.text if leitura else ""
                    ocr = leitura.confidence if leitura else 0.0
                    placa = normalize_plate(texto)
                    passou = det.confidence >= min_det and ocr >= min_ocr and placa.is_valid
                    no_gab = gabarito.get(placa.text, "")
                    cor = (VERDE if no_gab else VERMELHO) if passou else CINZA
                    if passou:
                        aceitas[placa.text] += 1

                    cv2.rectangle(vista, (b.x, b.y), (b.x + b.width, b.y + b.height), cor, 2)
                    escrever(vista, f"{texto or '?'} {no_gab}", (b.x, max(20, b.y - 28)), cor, 0.8)
                    escrever(vista, f"det {det.confidence:.2f} ocr {ocr:.2f} {b.width}x{b.height}px",
                             (b.x, max(40, b.y - 8)), cor, 0.5)
                    log.writerow([datetime.now().isoformat(timespec="milliseconds"), quadros, texto,
                                  placa.format or "", f"{det.confidence:.3f}", f"{ocr:.3f}", b.width,
                                  b.height, int(passou), no_gab])

                ms = (time.perf_counter() - inicio) * 1000
                lo, hi = (round(v * larg / DETECTOR_INPUT_WIDTH) for v in FAIXA_NA_ENTRADA)
                escrever(vista, f"{larg}x{alt} | {ms:.0f} ms/quadro | placa ideal ~{lo}-{hi} px de largura",
                         (10, 22), BRANCO)
                cv2.imshow(janela, vista)

                tecla = cv2.waitKey(1) & 0xFF
                if tecla in (ord("q"), 27):
                    break
                # Fechar no X também encerra (sem isto o loop seguia com a câmera ligada).
                if cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if tecla == ord("s"):
                    salvos += 1
                    destino = SAIDA / f"quadro_{carimbo}_{salvos:03d}.png"
                    cv2.imwrite(str(destino), vista)
                    print("Salvo:", destino)
        except KeyboardInterrupt:
            pass
        finally:
            fonte.stop()
            fonte.close()
            cv2.destroyAllWindows()

    print(f"\n{quadros} quadros analisados. Log: {arq_csv}")
    if aceitas:
        print("Leituras que o serviço aceitaria (texto: quadros):")
        for texto, n in aceitas.most_common():
            print(f"  {texto}: {n}  {'OK ' + gabarito[texto] if texto in gabarito else 'FORA DO GABARITO'}")
    if gabarito:
        faltaram = [f"{gid} {t}" for t, gid in gabarito.items() if t not in aceitas]
        print(f"Placas do gabarito nunca lidas: {', '.join(faltaram) or 'nenhuma'}")


if __name__ == "__main__":
    main()
