"""
app.py — Interface gráfica principal do Sistema SDR Inteligente.

Responsabilidade: construção da UI (PyQt6) e orquestração dos módulos.
Processamento pesado fica em:
    - src/dsp.py          → MotorDSP  (captura, demodulação, ring buffer)
    - src/transcricao.py  → TranscritorSDR  (Whisper STT + CSV)
    - src/analise.py      → CientistaSDR  (Llama 3.2 + dashboards)  [lazy import]
"""

from __future__ import annotations

import logging
import os
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPushButton,
    QScrollArea, QSlider, QTextEdit, QVBoxLayout, QWidget,
)

from config import (
    BASE_DIR,
    CAMINHO_BRUTOS,
    CAMINHO_CSV,
    FFT_SIZE,
    FREQUENCIA_PADRAO_MHZ,
    GANHO_PADRAO_DB,
    INTERVALO_TIMER_MS,
    LARGURA_PAINEL_LATERAL,
    LIMITE_ANALISE_REGISTOS,
    MAX_WORKERS_WHISPER,
    TAXA_AUDIO,
)
from src.dsp import MotorDSP
from src.transcricao import TranscritorSDR

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """
    Janela principal da aplicação.

    Delega:
        - Processamento DSP    → MotorDSP (src/dsp.py)
        - Transcrição de áudio → TranscritorSDR (src/transcricao.py)
        - Análise semântica    → CientistaSDR (src/analise.py, lazy import)
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SDR TCC — Monitorização e Edge AI")
        self.resize(1200, 850)
        self.setMinimumWidth(1000)

        # --- Estado da missão de captura ---
        self.gravando_ia            = False
        self.inicio_captura_tempo   = 0.0
        self.duracao_alvo_segundos  = 0.0
        self.horario_alvo           = ""

        # --- Caminhos configuráveis pelo utilizador ---
        self.pasta_destino_base   = CAMINHO_BRUTOS
        self.caminho_csv_analise  = CAMINHO_CSV
        os.makedirs(os.path.dirname(self.caminho_csv_analise), exist_ok=True)

        # --- Eixo de frequências pré-alocado (imutável) ---
        self._freq_axis = np.linspace(-0.512, 0.512, FFT_SIZE)

        # --- Motor DSP ---
        self.dsp = MotorDSP(
            frequencia_mhz=FREQUENCIA_PADRAO_MHZ,
            ganho_db=GANHO_PADRAO_DB,
        )
        self.dsp.on_ganho_real        = self._cb_ganho_real
        self.dsp.on_erro_antena       = self._cb_erro_antena
        self.dsp.on_chunk_pronto      = self._cb_chunk_pronto
        self.dsp.on_verificar_termino = self._cb_verificar_termino

        # --- Transcritor + pool de threads dedicado (limite explícito) ---
        self.transcritor  = TranscritorSDR(modelo_tamanho="base")
        self.pool_chunks  = ThreadPoolExecutor(
            max_workers=MAX_WORKERS_WHISPER,
            thread_name_prefix="whisper",
        )

        # --- Timer do gráfico de espectro ---
        self.timer_grafico = QTimer()
        self.timer_grafico.timeout.connect(self._atualizar_grafico)

        self._construir_interface()
        self.timer_grafico.start(INTERVALO_TIMER_MS)
        self.dsp.iniciar()

    # =========================================================================
    # CICLO DE VIDA DA JANELA
    # =========================================================================

    def closeEvent(self, event) -> None:
        """Encerramento gracioso: para captura, esgota pool, encerra motor DSP."""
        logger.info("A encerrar aplicação…")
        self.gravando_ia  = False
        self.dsp.gravando = False
        self.timer_grafico.stop()

        # Pausa mínima para o DSP worker perceber o sinal de paragem
        time.sleep(0.3)

        self.pool_chunks.shutdown(wait=False)
        self.dsp.parar()
        event.accept()

    # =========================================================================
    # CONSTRUÇÃO DA INTERFACE
    # =========================================================================

    def _construir_interface(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout_raiz = QHBoxLayout(central)

        # ---- Painel lateral com scroll ----
        scroll = QScrollArea()
        scroll.setFixedWidth(LARGURA_PAINEL_LATERAL)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        conteudo = QWidget()
        col = QVBoxLayout(conteudo)
        col.setSpacing(12)

        # -- Exibição da frequência sintonizada --
        self.lbl_freq = QLabel(f"{FREQUENCIA_PADRAO_MHZ} MHz")
        self.lbl_freq.setStyleSheet("font-size:40px;font-weight:bold;color:#00ffcc;")
        col.addWidget(self.lbl_freq, alignment=Qt.AlignmentFlag.AlignCenter)

        # -- Slider de frequência --
        col.addWidget(QLabel("📻 Frequência FM:"))
        self.slider_freq = QSlider(Qt.Orientation.Horizontal)
        self.slider_freq.setRange(875, 1080)
        self.slider_freq.setValue(int(FREQUENCIA_PADRAO_MHZ * 10))
        self.slider_freq.valueChanged.connect(self._mudar_frequencia)
        col.addWidget(self.slider_freq)

        # -- Slider de ganho --
        col.addWidget(QLabel("📡 Ganho de RF (Antena):"))
        self.lbl_ganho = QLabel(f"{GANHO_PADRAO_DB} dB")
        self.lbl_ganho.setStyleSheet("color:#ffaa00;font-weight:bold;")
        col.addWidget(self.lbl_ganho)
        self.slider_ganho = QSlider(Qt.Orientation.Horizontal)
        self.slider_ganho.setRange(0, 500)
        self.slider_ganho.setValue(int(GANHO_PADRAO_DB * 10))
        self.slider_ganho.valueChanged.connect(self._mudar_ganho)
        col.addWidget(self.slider_ganho)

        # -- Slider de volume --
        col.addWidget(QLabel("🔊 Volume:"))
        self.slider_volume = QSlider(Qt.Orientation.Horizontal)
        self.slider_volume.setRange(0, 200)
        self.slider_volume.setValue(100)
        self.slider_volume.valueChanged.connect(self._mudar_volume)
        col.addWidget(self.slider_volume)

        # -- Slider de altura do gráfico (eixo Y) --
        col.addWidget(QLabel("↕️ Altura do Gráfico (Y):"))
        self._altura_y = 0
        self.slider_y = QSlider(Qt.Orientation.Horizontal)
        self.slider_y.setRange(-150, 50)
        self.slider_y.setValue(self._altura_y)
        self.slider_y.valueChanged.connect(self._mudar_altura_y)
        col.addWidget(self.slider_y)

        # -- Slider de zoom horizontal (eixo X) --
        col.addWidget(QLabel("🔍 Zoom Visual (X):"))
        self._zoom = 1.0
        self.slider_zoom = QSlider(Qt.Orientation.Horizontal)
        self.slider_zoom.setRange(1, 10)
        self.slider_zoom.setValue(1)
        self.slider_zoom.valueChanged.connect(self._mudar_zoom)
        col.addWidget(self.slider_zoom)

        # -- Botão de áudio ao vivo --
        self.btn_audio = QPushButton("▶️ Ouvir Áudio ao Vivo")
        self.btn_audio.setStyleSheet("background-color:#28a745;color:white;height:35px;font-weight:bold;")
        self.btn_audio.clicked.connect(self._toggle_audio)
        col.addWidget(self.btn_audio)

        # -- Frame: configuração de captura --
        frame_cap = QFrame()
        frame_cap.setStyleSheet("background-color:#262626;border-radius:10px;padding:10px;")
        lay_cap = QVBoxLayout(frame_cap)
        lay_cap.addWidget(QLabel("📂 CONFIGURAÇÃO DE CAPTURA"))

        self.lbl_pasta = QLabel(f"Destino: .../{os.path.basename(self.pasta_destino_base)}")
        self.lbl_pasta.setStyleSheet("font-size:11px;color:#00ffcc;")
        btn_pasta = QPushButton("Alterar Pasta")
        btn_pasta.clicked.connect(self._escolher_pasta)
        lay_cap.addWidget(self.lbl_pasta)
        lay_cap.addWidget(btn_pasta)

        self.combo_modo = QComboBox()
        self.combo_modo.addItems([
            "Contínuo (Manual)",
            "Tempo Fixo (Minutos)",
            "Até Horário (HH:MM)",
        ])
        lay_cap.addWidget(self.combo_modo)

        self.input_param = QLineEdit()
        self.input_param.setPlaceholderText("Parâmetro…")
        self.input_param.setEnabled(False)
        self.combo_modo.currentIndexChanged.connect(
            lambda: self.input_param.setEnabled(self.combo_modo.currentIndex() > 0)
        )
        lay_cap.addWidget(self.input_param)

        self.btn_capturar = QPushButton("🔴 INICIAR MONITORAMENTO")
        self.btn_capturar.setStyleSheet(
            "background-color:#cc0000;color:white;font-weight:bold;height:40px;"
        )
        self.btn_capturar.clicked.connect(self._toggle_missao)
        lay_cap.addWidget(self.btn_capturar)
        col.addWidget(frame_cap)

        # -- Frame: análise semântica --
        frame_ana = QFrame()
        frame_ana.setStyleSheet(
            "background-color:#1e1e1e;border-radius:10px;border:1px solid #444;padding:10px;"
        )
        lay_ana = QVBoxLayout(frame_ana)
        lay_ana.addWidget(QLabel("📊 ANÁLISE SEMÂNTICA (IA)"))

        self.lbl_csv = QLabel(f"Arquivo: .../{os.path.basename(self.caminho_csv_analise)}")
        self.lbl_csv.setStyleSheet("font-size:10px;color:#6f42c1;")
        btn_csv = QPushButton("Escolher Banco (.csv)")
        btn_csv.clicked.connect(self._escolher_csv)
        lay_ana.addWidget(self.lbl_csv)
        lay_ana.addWidget(btn_csv)

        self.btn_analise = QPushButton("📈 GERAR DASHBOARD FINAL")
        self.btn_analise.setStyleSheet(
            "background-color:#6f42c1;color:white;font-weight:bold;height:40px;"
        )
        self.btn_analise.clicked.connect(self._abrir_analise)
        lay_ana.addWidget(self.btn_analise)
        col.addWidget(frame_ana)

        # -- Log de texto --
        self.caixa_texto = QTextEdit()
        self.caixa_texto.setReadOnly(True)
        col.addWidget(self.caixa_texto)

        scroll.setWidget(conteudo)
        layout_raiz.addWidget(scroll)

        # ---- Gráfico de espectro (bloqueado para o rato) ----
        pg.setConfigOption("background", "#151515")
        self.grafico = pg.PlotWidget(title="Espectro em Tempo Real")
        self.grafico.setYRange(self._altura_y - 60, self._altura_y + 40)
        self.grafico.showGrid(x=True, y=True, alpha=0.2)
        self.grafico.setMouseEnabled(x=False, y=False)
        self.grafico.hideButtons()

        self.curva_sinal  = self.grafico.plot(pen=pg.mkPen("#00ffcc", width=1.5))
        self.regiao_banda = pg.LinearRegionItem(
            values=[FREQUENCIA_PADRAO_MHZ - 0.085, FREQUENCIA_PADRAO_MHZ + 0.085],
            movable=False,
            brush=pg.mkBrush(255, 0, 0, 40),
        )
        self.grafico.addItem(self.regiao_banda)
        layout_raiz.addWidget(self.grafico, stretch=1)
        self._centralizar_grafico()

    # =========================================================================
    # CONTROLOS DA UI  (handlers dos sliders e botões)
    # =========================================================================

    def _mudar_frequencia(self, v: int) -> None:
        freq = round(v / 10.0, 1)
        self.dsp.set_frequencia(freq)
        self.lbl_freq.setText(f"{freq} MHz")
        self._centralizar_grafico()

    def _mudar_ganho(self, v: int) -> None:
        self.dsp.set_ganho(v / 10.0)

    def _mudar_volume(self, v: int) -> None:
        self.dsp.set_volume(v / 100.0)

    def _mudar_altura_y(self, v: int) -> None:
        self._altura_y = v
        self.grafico.setYRange(v - 60, v + 40)

    def _mudar_zoom(self, v: int) -> None:
        self._zoom = float(v)
        self._centralizar_grafico()

    def _toggle_audio(self) -> None:
        self.dsp.ouvindo_audio = not self.dsp.ouvindo_audio
        if self.dsp.ouvindo_audio:
            self.dsp.reset_ring_buffer()
            self.btn_audio.setText("⏹️ Parar Áudio")
        else:
            self.btn_audio.setText("▶️ Ouvir Áudio ao Vivo")

    def _centralizar_grafico(self) -> None:
        freq    = self.dsp.frequencia_mhz
        largura = 1.0 / self._zoom
        self.grafico.setXRange(freq - largura / 2, freq + largura / 2, padding=0)
        self.regiao_banda.setRegion([freq - 0.085, freq + 0.085])

    def _escolher_pasta(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Escolher Pasta para Capturas")
        if p:
            self.pasta_destino_base = p
            self.lbl_pasta.setText(f"Destino: .../{os.path.basename(p)}")

    def _escolher_csv(self) -> None:
        f, _ = QFileDialog.getOpenFileName(
            self, "Selecionar CSV de Dados",
            os.path.join(BASE_DIR, "dados"), "CSV (*.csv)",
        )
        if f:
            self.caminho_csv_analise = f
            self.lbl_csv.setText(f"Arquivo: .../{os.path.basename(f)}")

    # =========================================================================
    # GRÁFICO DE ESPECTRO
    # =========================================================================

    def _atualizar_grafico(self) -> None:
        dados = self.dsp.dados_grafico
        if dados is not None:
            psd = 10 * np.log10(
                np.abs(np.fft.fftshift(np.fft.fft(dados[:FFT_SIZE]))) ** 2 + 1e-12
            )
            f = self._freq_axis + self.dsp.frequencia_mhz
            self.curva_sinal.setData(f, psd)

    # =========================================================================
    # CALLBACKS DO MOTOR DSP  (chamados a partir de threads background)
    # Os lambdas garantem execução na thread da UI via QTimer.singleShot.
    # =========================================================================

    def _cb_ganho_real(self, ganho: float) -> None:
        QTimer.singleShot(0, lambda: self.lbl_ganho.setText(f"{ganho} dB"))

    def _cb_erro_antena(self, msg: str) -> None:
        QTimer.singleShot(0, lambda: self._log(f"❌ Erro Antena: {msg}"))

    def _cb_chunk_pronto(self, audio: np.ndarray) -> None:
        """Enviado pelo DSP worker quando um chunk de ~30 s está pronto."""
        self.pool_chunks.submit(self._processar_chunk, audio)

    def _cb_verificar_termino(self) -> None:
        QTimer.singleShot(0, self._verificar_termino)

    # =========================================================================
    # MISSÃO DE CAPTURA  (state machine)
    # =========================================================================

    def _toggle_missao(self) -> None:
        if self.gravando_ia:
            self._parar_missao()
        else:
            self._iniciar_missao()

    def _iniciar_missao(self) -> None:
        if not self.dsp.sdr:
            self._log("⚠️ A antena não está conectada.")
            return

        idx = self.combo_modo.currentIndex()
        if idx == 1:
            try:
                self.duracao_alvo_segundos = float(
                    self.input_param.text().replace(",", ".")
                ) * 60
            except ValueError:
                self._log("⚠️ Erro: Digite um número válido de minutos.")
                return
        elif idx == 2:
            self.horario_alvo = self.input_param.text().strip()
            if len(self.horario_alvo) != 5 or ":" not in self.horario_alvo:
                self._log("⚠️ Erro: Use o formato HH:MM (ex: 14:30)")
                return

        self.dsp.buffer_ia.clear()
        self.inicio_captura_tempo = time.time()
        self.gravando_ia  = True
        self.dsp.gravando = True
        self.btn_capturar.setText("⏹️ PARAR CAPTURA")
        self.btn_capturar.setStyleSheet(
            "background-color:#ff8c00;color:white;font-weight:bold;height:40px;"
        )
        self._log("🔴 A gravar áudio da antena em blocos de ~30 s…")

    def _parar_missao(self) -> None:
        self.gravando_ia  = False
        self.dsp.gravando = False
        self.btn_capturar.setText("🔴 INICIAR MONITORAMENTO")
        self.btn_capturar.setStyleSheet(
            "background-color:#cc0000;color:white;font-weight:bold;height:40px;"
        )
        self._log("⏹️ Monitoramento parado.")

    def _verificar_termino(self) -> None:
        """Verifica condições de paragem automática (chamado na thread da UI)."""
        idx = self.combo_modo.currentIndex()
        elapsed = time.time() - self.inicio_captura_tempo
        agora   = datetime.now().strftime("%H:%M")

        if idx == 1 and elapsed >= self.duracao_alvo_segundos:
            self._parar_missao()
        elif idx == 2 and agora >= self.horario_alvo:
            self._parar_missao()

    def _processar_chunk(self, audio: np.ndarray) -> None:
        """
        Salva chunk WAV e transcreve via Whisper.
        Executado no ThreadPoolExecutor — nunca toca em widgets diretamente.
        """
        ts     = datetime.now()
        data_s = ts.strftime("%Y-%m-%d")
        hora_s = ts.strftime("%Hh%Mm%Ss")
        pasta  = os.path.join(self.pasta_destino_base, data_s, hora_s)
        os.makedirs(pasta, exist_ok=True)
        path   = os.path.join(pasta, "chunk.wav")

        try:
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(TAXA_AUDIO)
                wf.writeframes((audio * 32767).astype(np.int16).tobytes())

            txt    = self.transcritor.transcrever(path, self.dsp.frequencia_mhz)
            resumo = (txt or "Nenhuma voz clara detectada.")[:60]
            QTimer.singleShot(0, lambda: self._log(f"✓ Chunk salvo ({hora_s}): {resumo}…"))

        except Exception:
            logger.exception("Erro ao processar chunk %s", path)
            QTimer.singleShot(0, lambda: self._log(f"❌ Erro ao salvar chunk ({hora_s})"))

    # =========================================================================
    # ANÁLISE SEMÂNTICA
    # =========================================================================

    def _abrir_analise(self) -> None:
        self.btn_analise.setEnabled(False)
        self._log("📊 A enviar dados para a IA Local…\nAguarde alguns instantes.")
        threading.Thread(target=self._rodar_analise, daemon=True).start()

    def _rodar_analise(self) -> None:
        try:
            from src.analise import CientistaSDR  # lazy import — evita ~200 MB no arranque
            cientista = CientistaSDR(caminho_csv=self.caminho_csv_analise)
            cientista.executar_analise(limite=LIMITE_ANALISE_REGISTOS)
            QTimer.singleShot(0, lambda: self._log("✅ Dashboard gerado! A pasta foi aberta."))
        except Exception as exc:
            logger.exception("Erro na análise semântica")
            QTimer.singleShot(0, lambda: self._log(f"❌ Erro na análise: {exc}"))
        finally:
            QTimer.singleShot(0, lambda: self.btn_analise.setEnabled(True))

    # =========================================================================
    # UTILITÁRIOS
    # =========================================================================

    def _log(self, msg: str) -> None:
        """Acrescenta uma linha ao log de texto e auto-scrolla para o fundo.
        Deve ser chamado SEMPRE na thread da UI."""
        atual = self.caixa_texto.toPlainText()
        self.caixa_texto.setText(f"{atual}\n{msg}" if atual else msg)
        sb = self.caixa_texto.verticalScrollBar()
        sb.setValue(sb.maximum())
