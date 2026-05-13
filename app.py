import sys
import os
import time
import threading
import wave
import queue
import numpy as np
import sounddevice as sd
import scipy.signal as signal
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

# --- CONFIGURAÇÃO DE CAMINHOS ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(BASE_DIR)
CAMINHO_DLL = os.path.join(BASE_DIR, 'ferramentas', 'rtl-sdr')
os.environ["PATH"] += os.pathsep + CAMINHO_DLL

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, QTextEdit, QFrame,
                             QFileDialog, QComboBox, QLineEdit, QScrollArea)
from PyQt6.QtCore import Qt, QTimer
import pyqtgraph as pg

from rtlsdr import RtlSdr
from src.transcricao import TranscritorSDR

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SDR - Analitico de Rádio com IA Local")
        self.resize(1200, 850)
        self.setMinimumWidth(1000)
        
        # --- VARIÁVEIS DE ESTADO ---
        self.frequencia_atual = 100.9  
        self.ganho_atual = 40.0        
        self.banda_atual = 170000.0  
        self.volume_atual = 1.0 
        self.zoom_visual = 1.0 
        self.altura_y_atual = 0 
        
        # --- GESTÃO DE PASTAS ---
        self.pasta_destino_base = os.path.join(BASE_DIR, 'dados', 'brutos')
        
        self.pasta_banco_dados = os.path.join(BASE_DIR, 'dados', 'banco_dados')
        os.makedirs(self.pasta_banco_dados, exist_ok=True)
        self.caminho_csv_analise = os.path.join(self.pasta_banco_dados, 'banco_transcricoes.csv')
        
        self.inicio_captura_tempo = 0
        self.duracao_alvo_segundos = 0
        self.horario_alvo = ""
        
        self.hardware_rodando = True 
        self.ouvindo_audio = False   
        self.gravando_ia = False 
        self.dados_grafico = None
        self.sdr = None
        self.stream_audio = None
        
        # --- RING BUFFER SPSC SEM LOCK (4 segundos @ 32kHz) ---
        # Single-Producer (thread DSP) / Single-Consumer (callback do driver de áudio)
        # Não requer lock: produtor escreve dados ANTES de avançar ring_write;
        # consumidor lê ring_write apenas uma vez por callback.
        self.TAMANHO_RING = 128000  # 4s @ 32kHz — margem contra underrun
        self.ring_buffer = np.zeros(self.TAMANHO_RING, dtype=np.float32)
        self.ring_write = 0   # escrito apenas pela thread DSP
        self.ring_read  = 0   # escrito apenas pelo callback do driver
        # ring_count não é mais necessário — o espaço disponível é derivado dos ponteiros
        self.lock_audio = threading.Lock()  # mantido apenas para reset no toggle_audio
        self.buffer_ia = []
        
        # --- POOL DE THREADS PARA TRANSCRIÇÃO (máx. 2 simultâneas) ---
        self.pool_chunks = ThreadPoolExecutor(max_workers=2, thread_name_prefix="whisper")
        
        # --- EIXO DE FREQUÊNCIAS PRÉ-ALOCADO (fixo, calculado uma vez) ---
        self._freq_axis = np.linspace(-0.512, 0.512, 4096)
        
        self.timer_grafico = QTimer()
        self.timer_grafico.timeout.connect(self.atualizar_grafico)
        self.transcritor = TranscritorSDR(modelo_tamanho="base")
        
        self.construir_interface()
        self.iniciar_hardware_background()

    def closeEvent(self, event):
        self.hardware_rodando = False
        self.gravando_ia = False
        
        # Espera o loop DSP terminar
        time.sleep(0.5)
        
        # Encerra pool de threads de transcrição
        if hasattr(self, 'pool_chunks'):
            self.pool_chunks.shutdown(wait=False)
        
        # Fecha stream de áudio
        if self.stream_audio:
            try:
                self.stream_audio.stop()
                self.stream_audio.close()
            except Exception:
                pass
        
        # Libera dispositivo SDR
        if self.sdr:
            try:
                self.sdr.close()
            except Exception:
                pass
        
        self.sdr = None
        self.stream_audio = None
        event.accept()

    # ==============================================================================
    # INTERFACE GRÁFICA (RESPONSIVA E COMPLETA)
    # ==============================================================================
    def construir_interface(self):
        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout_principal = QHBoxLayout(widget_central)

        # --- PAINEL LATERAL COM SCROLL ---
        scroll_area = QScrollArea()
        scroll_area.setFixedWidth(380)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        conteudo_lateral = QWidget()
        layout_controles = QVBoxLayout(conteudo_lateral)
        layout_controles.setSpacing(12) 
        
        # 1. BLOCO DE SINTONIA
        self.lbl_freq = QLabel(f"{self.frequencia_atual} MHz")
        self.lbl_freq.setStyleSheet("font-size: 40px; font-weight: bold; color: #00ffcc;")
        layout_controles.addWidget(self.lbl_freq, alignment=Qt.AlignmentFlag.AlignCenter)
        
        layout_controles.addWidget(QLabel("📻 Frequência FM:"))
        self.slider_freq = QSlider(Qt.Orientation.Horizontal)
        self.slider_freq.setRange(875, 1080); self.slider_freq.setValue(int(self.frequencia_atual * 10))
        self.slider_freq.valueChanged.connect(self.mudar_frequencia)
        layout_controles.addWidget(self.slider_freq)
        
        # LABEL E SLIDER DE GANHO
        layout_controles.addWidget(QLabel("📡 Ganho de RF (Antena):"))
        self.label_ganho = QLabel(f"{self.ganho_atual} dB")
        self.label_ganho.setStyleSheet("color: #ffaa00; font-weight: bold;")
        layout_controles.addWidget(self.label_ganho)
        
        self.slider_ganho = QSlider(Qt.Orientation.Horizontal)
        self.slider_ganho.setRange(0, 500); self.slider_ganho.setValue(int(self.ganho_atual * 10))
        self.slider_ganho.valueChanged.connect(self.mudar_ganho)
        layout_controles.addWidget(self.slider_ganho)

        layout_controles.addWidget(QLabel("🔊 Volume:"))
        self.slider_volume = QSlider(Qt.Orientation.Horizontal)
        self.slider_volume.setRange(0, 200); self.slider_volume.setValue(100)
        self.slider_volume.valueChanged.connect(self.mudar_volume)
        layout_controles.addWidget(self.slider_volume)
        
        layout_controles.addWidget(QLabel("↕️ Altura do Gráfico (Y):"))
        self.slider_y = QSlider(Qt.Orientation.Horizontal)
        self.slider_y.setRange(-150, 50); self.slider_y.setValue(self.altura_y_atual)
        self.slider_y.valueChanged.connect(self.mudar_altura_grafico)
        layout_controles.addWidget(self.slider_y)

        layout_controles.addWidget(QLabel("🔍 Zoom Visual (X):"))
        self.slider_zoom = QSlider(Qt.Orientation.Horizontal)
        self.slider_zoom.setRange(1, 10); self.slider_zoom.setValue(int(self.zoom_visual))
        self.slider_zoom.valueChanged.connect(self.mudar_zoom)
        layout_controles.addWidget(self.slider_zoom)
        
        self.btn_audio = QPushButton("▶️ Ouvir Áudio ao Vivo")
        self.btn_audio.setStyleSheet("background-color: #28a745; color: white; height: 35px; font-weight: bold;")
        self.btn_audio.clicked.connect(self.toggle_audio)
        layout_controles.addWidget(self.btn_audio)

        # 2. PAINEL DE CAPTURA
        frame_cap = QFrame()
        frame_cap.setStyleSheet("background-color: #262626; border-radius: 10px; padding: 10px;")
        lay_cap = QVBoxLayout(frame_cap)
        lay_cap.addWidget(QLabel("📂 CONFIGURAÇÃO DE CAPTURA"))
        
        self.lbl_pasta = QLabel(f"Destino: .../{os.path.basename(self.pasta_destino_base)}")
        self.lbl_pasta.setStyleSheet("font-size: 11px; color: #00ffcc;")
        btn_pasta = QPushButton("Alterar Pasta")
        btn_pasta.clicked.connect(self.escolher_pasta)
        
        lay_cap.addWidget(self.lbl_pasta)
        lay_cap.addWidget(btn_pasta)
        
        self.combo_modo = QComboBox()
        self.combo_modo.addItems(["Contínuo (Manual)", "Tempo Fixo (Minutos)", "Até Horário (HH:MM)"])
        lay_cap.addWidget(self.combo_modo)
        
        self.input_param = QLineEdit(); self.input_param.setPlaceholderText("Parâmetro..."); self.input_param.setEnabled(False)
        self.combo_modo.currentIndexChanged.connect(lambda: self.input_param.setEnabled(self.combo_modo.currentIndex() > 0))
        lay_cap.addWidget(self.input_param)
        
        self.btn_capturar = QPushButton("🔴 INICIAR MONITORAMENTO")
        self.btn_capturar.setStyleSheet("background-color: #cc0000; color: white; font-weight: bold; height: 40px;")
        self.btn_capturar.clicked.connect(self.toggle_missao)
        lay_cap.addWidget(self.btn_capturar)
        layout_controles.addWidget(frame_cap)

        # 3. PAINEL DE ANÁLISE 
        frame_ana = QFrame()
        frame_ana.setStyleSheet("background-color: #1e1e1e; border-radius: 10px; border: 1px solid #444; padding: 10px;")
        lay_ana = QVBoxLayout(frame_ana)
        lay_ana.addWidget(QLabel("📊 ANÁLISE SEMÂNTICA (IA)"))
        
        self.lbl_csv = QLabel(f"Arquivo: .../{os.path.basename(os.path.dirname(self.caminho_csv_analise))}/{os.path.basename(self.caminho_csv_analise)}")
        self.lbl_csv.setStyleSheet("font-size: 10px; color: #6f42c1;")
        btn_csv = QPushButton("Escolher Banco (.csv)")
        btn_csv.clicked.connect(self.escolher_csv)
        
        lay_ana.addWidget(self.lbl_csv)
        lay_ana.addWidget(btn_csv)

        self.btn_analise = QPushButton("📈 GERAR DASHBOARD FINAL")
        self.btn_analise.setStyleSheet("background-color: #6f42c1; color: white; font-weight: bold; height: 40px;")
        self.btn_analise.clicked.connect(self.abrir_analise)
        lay_ana.addWidget(self.btn_analise)
        layout_controles.addWidget(frame_ana)
        
        self.caixa_texto = QTextEdit(); self.caixa_texto.setReadOnly(True)
        layout_controles.addWidget(self.caixa_texto)
        
        scroll_area.setWidget(conteudo_lateral)
        layout_principal.addWidget(scroll_area)
        
        # --- GRÁFICO (BLOQUEADO PARA O RATO) ---
        pg.setConfigOption('background', '#151515')
        self.grafico = pg.PlotWidget(title="Espectro em Tempo Real")
        self.grafico.setYRange(self.altura_y_atual - 60, self.altura_y_atual + 40)
        self.grafico.showGrid(x=True, y=True, alpha=0.2)
        
        self.grafico.setMouseEnabled(x=False, y=False)
        self.grafico.hideButtons()
        
        self.curva_sinal = self.grafico.plot(pen=pg.mkPen('#00ffcc', width=1.5))
        
        self.regiao_banda = pg.LinearRegionItem(values=[100.8, 101.0], movable=False, brush=pg.mkBrush(255, 0, 0, 40))
        self.grafico.addItem(self.regiao_banda)
        layout_principal.addWidget(self.grafico, stretch=1)
        self.centralizar_grafico()

    # ==============================================================================
    # LÓGICA DE SELEÇÃO E CONTROLES 
    # ==============================================================================
    def escolher_pasta(self):
        p = QFileDialog.getExistingDirectory(self, "Escolher Pasta para Capturas")
        if p: 
            self.pasta_destino_base = p
            self.lbl_pasta.setText(f"Destino: .../{os.path.basename(p)}")

    def escolher_csv(self):
        f, _ = QFileDialog.getOpenFileName(self, "Selecionar CSV de Dados", os.path.join(BASE_DIR, 'dados'), "CSV (*.csv)")
        if f: 
            self.caminho_csv_analise = f
            self.lbl_csv.setText(f"Arquivo: .../{os.path.basename(os.path.dirname(f))}/{os.path.basename(f)}")

    def centralizar_grafico(self):
        largura = 1.0 / self.zoom_visual 
        self.grafico.setXRange(self.frequencia_atual - (largura / 2), self.frequencia_atual + (largura / 2), padding=0)
        self.regiao_banda.setRegion([self.frequencia_atual - 0.085, self.frequencia_atual + 0.085])

    def mudar_frequencia(self, v):
        self.frequencia_atual = round(v / 10.0, 2)
        self.lbl_freq.setText(f"{self.frequencia_atual} MHz")
        if self.sdr: self.sdr.center_freq = self.frequencia_atual * 1e6
        self.centralizar_grafico()

    def mudar_zoom(self, v):
        self.zoom_visual = float(v)
        self.centralizar_grafico()

    def mudar_altura_grafico(self, v):
        self.altura_y_atual = v
        self.grafico.setYRange(self.altura_y_atual - 60, self.altura_y_atual + 40)

    def mudar_ganho(self, v):
        """O Algoritmo de Ganho de Hardware Inteligente"""
        self.ganho_atual = v / 10.0
        
        if self.sdr: 
            try:
                # 1. Pede à antena a lista dos valores de hardware que ela aceita
                ganhos_validos = self.sdr.valid_gains_db
                
                # 2. Matemáticamente, encontra o degrau mais próximo
                ganho_real = min(ganhos_validos, key=lambda x: abs(x - self.ganho_atual))
                
                # 3. Força a antena a obedecer e aplica o ganho
                self.sdr.set_manual_gain_enabled(True)
                self.sdr.gain = ganho_real
                
                # 4. Atualiza a UI para o Engenheiro saber o ganho exato que a antena está a usar!
                self.label_ganho.setText(f"{ganho_real} dB")
            except Exception as e:
                pass
        else:
            self.label_ganho.setText(f"{self.ganho_atual} dB")

    def mudar_volume(self, v):
        self.volume_atual = v / 100.0

    def toggle_audio(self):
        self.ouvindo_audio = not self.ouvindo_audio
        self.btn_audio.setText("⏹️ Parar Áudio" if self.ouvindo_audio else "▶️ Ouvir Áudio ao Vivo")
        if self.ouvindo_audio:
            # Reset atômico: para o consumidor brevemente enquanto reinicia ponteiros
            with self.lock_audio:
                self.ring_buffer.fill(0)
                self.ring_write = 0
                self.ring_read  = 0

    # ==============================================================================
    # SDR E PROCESSAMENTO (O MOTOR RÁPIDO E FLUIDO)
    # ==============================================================================
    def callback_audio(self, out, frames, time_info, status):
        # SPSC lock-free: lê ring_write uma vez (snapshot), nunca bloqueia
        wr = self.ring_write
        rd = self.ring_read
        disponivel = (wr - rd) % self.TAMANHO_RING
        if disponivel >= frames:
            end = rd + frames
            if end <= self.TAMANHO_RING:
                out[:, 0] = self.ring_buffer[rd:end]
            else:
                parte1 = self.TAMANHO_RING - rd
                out[:parte1, 0] = self.ring_buffer[rd:]
                out[parte1:frames, 0] = self.ring_buffer[:frames - parte1]
            # Avança o ponteiro de leitura APÓS copiar os dados
            self.ring_read = (rd + frames) % self.TAMANHO_RING
        else:
            # Buffer underrun — silêncio; log para diagnóstico
            out.fill(0)
            if status:
                print(f"[AUDIO] underrun: disponível={disponivel} frames={frames}")

    def iniciar_hardware_background(self):
        self.timer_grafico.start(120)  # ~8 FPS — perceptualmente idêntico para espectro
        # Fila de IQ bruto entre a thread SDR (bloqueante USB) e a thread DSP
        self._iq_queue = queue.Queue(maxsize=4)  # no máximo 4 blocos pendentes
        threading.Thread(target=self._thread_sdr_reader, daemon=True).start()
        threading.Thread(target=self._thread_dsp_worker, daemon=True).start()

    # --------------------------------------------------------------------------
    # THREAD 1 — Leitura SDR (bloqueante USB, ~250ms por ciclo)
    # Única responsabilidade: ler amostras e empurrar para a fila IQ.
    # Nunca faz DSP. Assim o DSP nunca fica parado esperando o USB.
    # --------------------------------------------------------------------------
    def _thread_sdr_reader(self):
        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = 1024000
            self.sdr.center_freq = self.frequencia_atual * 1e6

            try:
                self.sdr.set_manual_gain_enabled(True)
                ganhos_validos = self.sdr.valid_gains_db
                ganho_real = min(ganhos_validos, key=lambda x: abs(x - self.ganho_atual))
                self.sdr.gain = ganho_real
                QTimer.singleShot(0, lambda: self.label_ganho.setText(f"{ganho_real} dB"))
            except:
                pass

            # 256k samples @ 1024 kS/s → ~250ms de bloqueio USB por leitura
            # Produz 256k/4/8 = 8000 samples de áudio @ 32kHz = 250ms de áudio
            # Taxa de produção == taxa de consumo → buffer estável
            TAMANHO_BLOCO_SDR = 262144  # múltiplo de 16384 (exigido pelo rtl-sdr)

            while self.hardware_rodando:
                if not self.sdr:
                    break
                try:
                    amostras = self.sdr.read_samples(TAMANHO_BLOCO_SDR)
                    # Expõe para o gráfico (sem cópia — apenas referência)
                    self.dados_grafico = amostras
                    # Envia para DSP; descarta se a fila estiver cheia (evita acúmulo)
                    try:
                        self._iq_queue.put_nowait(amostras)
                    except queue.Full:
                        pass  # DSP não está a acompanhar; descarta bloco antigo
                except Exception:
                    pass
        except Exception as e:
            QTimer.singleShot(0, lambda: self.caixa_texto.setText(f"Erro Antena: {e}"))

    # --------------------------------------------------------------------------
    # THREAD 2 — DSP Worker (filtragem + demodulação + escrita no ring buffer)
    # Consome amostras IQ da fila, nunca bloqueia em USB.
    # --------------------------------------------------------------------------
    def _thread_dsp_worker(self):
        # Aguarda o SDR inicializar (stream de áudio precisa de taxa_audio)
        taxa_original  = 1024000
        decimacao_iq   = 4
        decimacao_audio = 8
        taxa_audio = (taxa_original // decimacao_iq) // decimacao_audio  # 32000 Hz

        dt    = 1.0 / taxa_audio
        alpha = dt / (75e-6 + dt)
        b_deemp = [alpha]
        a_deemp = [1.0, -(1.0 - alpha)]
        zi_deemp = signal.lfilter_zi(b_deemp, a_deemp) * 0.0

        banda_filtro_memoria = 0
        b_band, a_band, zi_band = None, None, None
        _prev_iq = np.array([0j], dtype=complex)

        # Inicia o stream de áudio aqui (mesma thread que vai escrever no ring)
        self.stream_audio = sd.OutputStream(
            samplerate=taxa_audio,
            channels=1,
            dtype='float32',
            blocksize=2048,   # ~64ms @ 32kHz — equilíbrio entre latência e estabilidade
            callback=self.callback_audio,
            latency='high',   # permite ao driver usar buffers internos maiores
        )
        self.stream_audio.start()

        while self.hardware_rodando:
            try:
                # Bloqueia no máximo 1s; se não vier amostra, volta ao topo
                amostras = self._iq_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                if not (self.ouvindo_audio or self.gravando_ia):
                    continue

                iq_256k = amostras[::decimacao_iq]

                if self.banda_atual != banda_filtro_memoria:
                    banda_filtro_memoria = self.banda_atual
                    cutoff = min(banda_filtro_memoria / 2.0, 127000.0)
                    b_band, a_band = signal.butter(3, cutoff / 128000.0, btype='low')
                    zi_band = signal.lfilter_zi(b_band, a_band) * iq_256k[0]

                iq_filtrado, zi_band = signal.lfilter(b_band, a_band, iq_256k, zi=zi_band)

                iq_completo = np.concatenate((_prev_iq, iq_filtrado))
                _prev_iq[0]  = iq_filtrado[-1]
                demodulado   = np.angle(iq_completo[1:] * np.conj(iq_completo[:-1]))

                audio_cru = demodulado[::decimacao_audio]
                audio_filtrado_final, zi_deemp = signal.lfilter(b_deemp, a_deemp, audio_cru, zi=zi_deemp)
                audio_final = (audio_filtrado_final * self.volume_atual * 0.5).astype(np.float32)

                if self.ouvindo_audio:
                    n  = len(audio_final)
                    wr = self.ring_write
                    # Calcula espaço livre ANTES de escrever
                    rd = self.ring_read
                    espaco_livre = (rd - wr - 1) % self.TAMANHO_RING
                    if n > espaco_livre:
                        # Overflow: avança read para abrir espaço (descarta mais antigo)
                        excesso = n - espaco_livre
                        self.ring_read = (rd + excesso) % self.TAMANHO_RING
                    # Escreve dados no ring ANTES de atualizar ring_write
                    end = wr + n
                    if end <= self.TAMANHO_RING:
                        self.ring_buffer[wr:end] = audio_final
                    else:
                        parte1 = self.TAMANHO_RING - wr
                        self.ring_buffer[wr:]      = audio_final[:parte1]
                        self.ring_buffer[:n-parte1] = audio_final[parte1:]
                    # Publica o novo ponteiro — o callback verá dados válidos
                    self.ring_write = (wr + n) % self.TAMANHO_RING

                if self.gravando_ia:
                    self.buffer_ia.append(audio_final)
                    if len(self.buffer_ia) >= 300:
                        fatia = np.concatenate(self.buffer_ia[:300])
                        self.buffer_ia = self.buffer_ia[300:]
                        self.pool_chunks.submit(self.processar_chunk, fatia)
                        self.verificar_termino()
            except Exception:
                pass

    def atualizar_grafico(self):
        if self.dados_grafico is not None:
            psd = 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(self.dados_grafico[:4096])))**2)
            f = self._freq_axis + self.frequencia_atual
            self.curva_sinal.setData(f, psd)

    # ==============================================================================
    # MISSÃO DE CAPTURA
    # ==============================================================================
    def toggle_missao(self):
        if self.gravando_ia:
            self.gravando_ia = False
            self.btn_capturar.setText("🔴 INICIAR MONITORAMENTO")
            self.btn_capturar.setStyleSheet("background-color: #cc0000; color: white; font-weight: bold; height: 45px;")
            self.caixa_texto.setText(self.caixa_texto.toPlainText() + "\n⏹️ Monitoramento Manual Parado.")
        else:
            if not self.sdr:
                self.caixa_texto.setText("⚠️ A antena não está conectada.")
                return

            idx = self.combo_modo.currentIndex()
            if idx == 1: 
                try:
                    self.duracao_alvo_segundos = float(self.input_param.text().replace(',', '.')) * 60
                except:
                    self.caixa_texto.setText("⚠️ Erro: Digite um número válido de minutos.")
                    return
            elif idx == 2: 
                self.horario_alvo = self.input_param.text().strip()
                if len(self.horario_alvo) != 5 or ":" not in self.horario_alvo:
                    self.caixa_texto.setText("⚠️ Erro: Use o formato HH:MM (ex: 14:30)")
                    return
            
            self.buffer_ia = []
            self.inicio_captura_tempo = time.time()
            self.gravando_ia = True
            self.btn_capturar.setText("⏹️ PARAR CAPTURA")
            self.btn_capturar.setStyleSheet("background-color: #ff8c00; color: white; font-weight: bold; height: 45px;")
            self.caixa_texto.setText("A gravar áudio da antena em blocos de 30s...")

    def verificar_termino(self):
        idx = self.combo_modo.currentIndex()
        if idx == 1 and (time.time() - self.inicio_captura_tempo >= self.duracao_alvo_segundos):
            QTimer.singleShot(0, self.toggle_missao)
        elif idx == 2 and (datetime.now().strftime("%H:%M") >= self.horario_alvo):
            QTimer.singleShot(0, self.toggle_missao)

    def processar_chunk(self, audio):
        data_s = datetime.now().strftime("%Y-%m-%d"); hora_s = datetime.now().strftime("%Hh%Mm%Ss")
        pasta = os.path.join(self.pasta_destino_base, data_s, hora_s); os.makedirs(pasta, exist_ok=True)
        path = os.path.join(pasta, "chunk.wav")
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(32000)
            wf.writeframes((audio * 32767).astype(np.int16).tobytes())
        
        txt = self.transcritor.transcrever(path, self.frequencia_atual)
        if not txt: txt = "Nenhuma voz clara detectada."
        QTimer.singleShot(0, lambda: self.caixa_texto.setText(self.caixa_texto.toPlainText() + f"\n✓ Chunk salvo ({hora_s}): {txt[:50]}..."))

    # ==============================================================================
    # BOTÃO DO ANALISADOR E GRÁFICOS
    # ==============================================================================
    def abrir_analise(self):
        self.btn_analise.setEnabled(False)
        self.caixa_texto.setText(self.caixa_texto.toPlainText() + "\n📊 A enviar dados para a IA Local...\nAguarde alguns instantes.")
        threading.Thread(target=self.rodar_analise, daemon=True).start()

    def rodar_analise(self):
        try:
            from src.analise import CientistaSDR
            cientista = CientistaSDR(caminho_csv=self.caminho_csv_analise)
            cientista.executar_analise(limite=5)
            QTimer.singleShot(0, lambda: self.caixa_texto.setText(self.caixa_texto.toPlainText() + "\n✅ Dashboard gerado! A pasta foi aberta."))
        except Exception as e:
            QTimer.singleShot(0, lambda: self.caixa_texto.setText(self.caixa_texto.toPlainText() + f"\n❌ Erro na análise: {e}"))
        finally:
            QTimer.singleShot(0, lambda: self.btn_analise.setEnabled(True))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec())