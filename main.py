"""
main.py — Entry point do Sistema SDR Inteligente.

Execute para iniciar a aplicação:
    python main.py

Responsabilidade deste ficheiro: apenas bootstrap.
- Configura o PATH para as DLLs do RTL-SDR.
- Configura o sistema de logging.
- Cria QApplication e MainWindow.
- Não contém lógica de negócio.
"""

import logging
import os
import sys

# Força UTF-8 no stdout/stderr para suportar emojis no terminal do Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Garante que o diretório raiz está no PYTHONPATH independentemente de
# onde o utilizador invoca o script.
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Configura o PATH para as DLLs nativas ANTES de qualquer import PyQt / rtlsdr.
from config import CAMINHO_DLL  # noqa: E402  (import após manipulação de sys.path)
os.environ["PATH"] += os.pathsep + CAMINHO_DLL

# Garante que o ffmpeg está no PATH do processo (Whisper depende dele).
# Procura pelo bin/ do ffmpeg instalado via winget de forma dinâmica.
_FFMPEG_BASE = os.path.join(
    os.environ.get("LOCALAPPDATA", ""),
    "Microsoft", "WinGet", "Packages",
)
if os.path.isdir(_FFMPEG_BASE):
    for _pkg in os.listdir(_FFMPEG_BASE):
        if _pkg.startswith("Gyan.FFmpeg"):
            _pkg_path = os.path.join(_FFMPEG_BASE, _pkg)
            # Procura pela pasta bin/ dentro do pacote
            for _root, _dirs, _files in os.walk(_pkg_path):
                if "ffmpeg.exe" in _files:
                    if _root not in os.environ["PATH"]:
                        os.environ["PATH"] = _root + os.pathsep + os.environ["PATH"]
                    break
            break

# Configura logging estruturado para toda a aplicação.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from PyQt6.QtWidgets import QApplication  # noqa: E402
from app import MainWindow                 # noqa: E402


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("SDR TCC — Monitorização e Edge AI")

    win = MainWindow()
    win.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
