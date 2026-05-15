"""
src/dsp.py — Motor DSP do Sistema SDR Inteligente.

Responsabilidade: leitura de amostras I/Q do hardware RTL-SDR,
desmodulação FM e gestão do ring buffer de áudio.

Pipeline de dados:
    RTL-SDR (USB) ──► Fila IQ ──► DSP (decimação + filtros) ──► Ring Buffer ──► sounddevice

Threads internas:
    - sdr-reader : leitura bloqueante USB (~250 ms/ciclo). Nunca faz DSP.
    - dsp-worker : filtragem, demodulação, escrita no ring buffer e acumulação para IA.

Padrão de buffer:
    SPSC (Single-Producer / Single-Consumer) lock-free.
    O produtor (dsp-worker) escreve ANTES de avançar ring_write.
    O consumidor (callback_audio) lê ring_write apenas uma vez por callback.
    O lock_reset é utilizado APENAS no reset explícito dos ponteiros (toggle de áudio).
"""

from __future__ import annotations

import time

import logging
import queue
import threading
from typing import Callable

import numpy as np
import scipy.signal as signal
import sounddevice as sd
from rtlsdr import RtlSdr

from config import (
    BANDA_PADRAO_HZ,
    BLOCKSIZE_AUDIO,
    BLOCOS_POR_CHUNK,
    DECIMACAO_AUDIO,
    DECIMACAO_IQ,
    DEEMPHASIS_TAU,
    FREQUENCIA_PADRAO_MHZ,
    GANHO_PADRAO_DB,
    ORDEM_FILTRO,
    SAMPLE_RATE_SDR,
    TAMANHO_BLOCO_SDR,
    TAMANHO_RING,
    TAXA_AUDIO,
)

logger = logging.getLogger(__name__)

# Constante interna: frequência de corte normalizada do filtro de canal.
# O sinal IQ após decimação de 4× tem taxa de Nyquist de 128 kHz.
_NYQUIST_POS_IQ = 128_000.0


class MotorDSP:
    """
    Encapsula todo o pipeline de processamento de sinal digital.

    Uso:
        motor = MotorDSP(frequencia_mhz=100.9, ganho_db=40.0)
        motor.on_ganho_real      = lambda g: ...   # UI callback
        motor.on_erro_antena     = lambda m: ...   # UI callback
        motor.on_chunk_pronto    = lambda a: ...   # chamado com ndarray de áudio
        motor.on_verificar_termino = lambda: ...   # verifica condições de paragem
        motor.iniciar()
        ...
        motor.parar()
    """

    def __init__(self, frequencia_mhz: float = FREQUENCIA_PADRAO_MHZ,
                 ganho_db: float = GANHO_PADRAO_DB) -> None:

        self.frequencia_mhz = frequencia_mhz
        self.ganho_db       = ganho_db
        self.banda_hz       = BANDA_PADRAO_HZ
        self.volume         = 1.0

        # --- Flags de controlo ---
        self._rodando     = False
        self.ouvindo_audio = False
        self.gravando      = False

        # --- Hardware e stream ---
        self.sdr: RtlSdr | None         = None
        self.stream_audio: sd.OutputStream | None = None

        # --- Fila IQ entre thread SDR e thread DSP ---
        self._iq_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=4)

        # --- Ring Buffer SPSC ---
        self.ring_buffer = np.zeros(TAMANHO_RING, dtype=np.float32)
        self.ring_write  = 0
        self.ring_read   = 0
        self._lock_reset = threading.Lock()  # protege APENAS o reset dos ponteiros

        # --- Dado exposto à UI para o gráfico de espectro ---
        self.dados_grafico: np.ndarray | None = None

        # --- Buffer de acumulação para IA ---
        self.buffer_ia: list[np.ndarray] = []

        # --- Callbacks configurados externamente pela MainWindow ---
        self.on_ganho_real:        Callable[[float], None] | None = None
        self.on_erro_antena:       Callable[[str], None]   | None = None
        self.on_chunk_pronto:      Callable[[np.ndarray], None] | None = None
        self.on_verificar_termino: Callable[[], None]      | None = None

    # =========================================================================
    # API PÚBLICA
    # =========================================================================

    def iniciar(self) -> None:
        """Inicia as duas threads de background (SDR reader + DSP worker)."""
        self._rodando = True
        threading.Thread(
            target=self._thread_sdr_reader,
            daemon=True,
            name="sdr-reader",
        ).start()
        threading.Thread(
            target=self._thread_dsp_worker,
            daemon=True,
            name="dsp-worker",
        ).start()
        logger.info("MotorDSP iniciado (%.1f MHz, %.1f dB)", self.frequencia_mhz, self.ganho_db)

    def parar(self) -> None:
        """Sinaliza encerramento e libera recursos de hardware."""
        self._rodando = False
        self._fechar_stream_audio()
        self._fechar_sdr()
        logger.info("MotorDSP encerrado")

    def set_frequencia(self, mhz: float) -> None:
        self.frequencia_mhz = mhz
        if self.sdr:
            try:
                self.sdr.center_freq = mhz * 1e6
            except Exception:
                logger.warning("Falha ao alterar frequência para %.1f MHz", mhz)

    def set_ganho(self, db: float) -> None:
        """Aplica smart snapping: mapeia valor contínuo para o degrau discreto do hardware."""
        self.ganho_db = db
        if not self.sdr:
            return
        try:
            validos = self.sdr.valid_gains_db
            real = min(validos, key=lambda x: abs(x - db))
            self.sdr.set_manual_gain_enabled(True)
            self.sdr.gain = real
            if self.on_ganho_real:
                self.on_ganho_real(real)
        except Exception:
            logger.warning("Falha ao aplicar ganho %.1f dB", db)

    def set_volume(self, v: float) -> None:
        self.volume = v

    def reset_ring_buffer(self) -> None:
        """Reset atômico do ring buffer. Chamado ao (re)ativar áudio ao vivo."""
        with self._lock_reset:
            self.ring_buffer.fill(0)
            self.ring_write = 0
            self.ring_read  = 0

    # =========================================================================
    # CALLBACK DE ÁUDIO  (thread do driver de áudio — sounddevice)
    # =========================================================================

    def callback_audio(self, out: np.ndarray, frames: int, time_info, status) -> None:
        """
        Alimenta o output stream com amostras do ring buffer (SPSC lock-free).
        Em caso de underrun, envia silêncio e regista aviso de diagnóstico.
        """
        wr = self.ring_write       # snapshot único — produtor pode avançar depois
        rd = self.ring_read
        disponivel = (wr - rd) % TAMANHO_RING

        if disponivel >= frames:
            end = rd + frames
            if end <= TAMANHO_RING:
                out[:, 0] = self.ring_buffer[rd:end]
            else:
                parte1 = TAMANHO_RING - rd
                out[:parte1, 0]  = self.ring_buffer[rd:]
                out[parte1:, 0]  = self.ring_buffer[:frames - parte1]
            self.ring_read = (rd + frames) % TAMANHO_RING
        else:
            out.fill(0)
            if status:
                logger.debug("Audio underrun: disponível=%d | pedido=%d", disponivel, frames)

    # =========================================================================
    # THREAD 1 — SDR Reader  (bloqueante USB, ~250 ms/ciclo)
    # Única responsabilidade: ler amostras e empurrar para a fila IQ.
    # Nunca executa DSP — assim o DSP nunca fica parado à espera do USB.
    # =========================================================================

    def _thread_sdr_reader(self) -> None:
        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = SAMPLE_RATE_SDR
            self.sdr.center_freq = self.frequencia_mhz * 1e6
            self._aplicar_ganho_inicial()
            logger.info("RTL-SDR inicializado: %.3f MHz, %d kS/s",
                        self.frequencia_mhz, SAMPLE_RATE_SDR // 1000)

            erros_consecutivos = 0
            MAX_ERROS_CONSECUTIVOS = 5

            while self._rodando:
                try:
                    amostras = self.sdr.read_samples(TAMANHO_BLOCO_SDR)
                    erros_consecutivos = 0            # leitura OK — reset contador
                    self.dados_grafico = amostras     # referência — sem cópia
                    try:
                        self._iq_queue.put_nowait(amostras)
                    except queue.Full:
                        logger.debug("Fila IQ cheia — bloco descartado")

                except OSError as exc:
                    # OSError geralmente indica que o dispositivo foi desconectado
                    # ou que o ponteiro nativo é inválido — erro irrecuperável.
                    logger.error(
                        "Erro de hardware SDR irrecuperável (dispositivo desconectado?): %s", exc
                    )
                    self._rodando = False
                    if self.on_erro_antena:
                        self.on_erro_antena(
                            f"Hardware SDR desconectado ou inválido: {exc}"
                        )
                    break

                except Exception:
                    erros_consecutivos += 1
                    logger.exception(
                        "Erro na leitura SDR (%d/%d); a tentar novamente…",
                        erros_consecutivos,
                        MAX_ERROS_CONSECUTIVOS,
                    )
                    if erros_consecutivos >= MAX_ERROS_CONSECUTIVOS:
                        logger.error(
                            "Número máximo de erros consecutivos atingido — a encerrar thread SDR."
                        )
                        self._rodando = False
                        if self.on_erro_antena:
                            self.on_erro_antena(
                                "Demasiados erros consecutivos na leitura SDR. "
                                "Verifique o hardware."
                            )
                        break
                    time.sleep(0.5)  # aguarda antes de retentar

        except Exception as exc:
            logger.error("Falha crítica ao iniciar hardware SDR: %s", exc)
            if self.on_erro_antena:
                self.on_erro_antena(str(exc))

    def _aplicar_ganho_inicial(self) -> None:
        try:
            validos = self.sdr.valid_gains_db
            real = min(validos, key=lambda x: abs(x - self.ganho_db))
            self.sdr.set_manual_gain_enabled(True)
            self.sdr.gain = real
            if self.on_ganho_real:
                self.on_ganho_real(real)
            logger.info("Ganho inicial aplicado: %.1f dB (pedido: %.1f dB)", real, self.ganho_db)
        except Exception:
            logger.warning("Não foi possível aplicar ganho inicial — a usar ganho automático")

    # =========================================================================
    # THREAD 2 — DSP Worker  (filtragem + demodulação + ring buffer)
    # Consome amostras IQ da fila; nunca bloqueia em USB.
    # O estado dos filtros (zi) é preservado entre iterações para eliminar
    # artefactos de fronteira (cliques e descontinuidades).
    # =========================================================================

    def _thread_dsp_worker(self) -> None:
        # --- Estado persistente dos filtros ---
        dt    = 1.0 / TAXA_AUDIO
        alpha = dt / (DEEMPHASIS_TAU + dt)
        b_deemp = np.array([alpha])
        a_deemp = np.array([1.0, -(1.0 - alpha)])
        zi_deemp = signal.lfilter_zi(b_deemp, a_deemp) * 0.0

        banda_cache: float = 0.0
        b_band = a_band = zi_band = None
        prev_iq = np.array([0j], dtype=complex)

        # Inicia o stream de áudio nesta thread (produtora do ring buffer).
        # O stream fica ativo enquanto a thread existir.
        self.stream_audio = sd.OutputStream(
            samplerate=TAXA_AUDIO,
            channels=1,
            dtype="float32",
            blocksize=BLOCKSIZE_AUDIO,
            callback=self.callback_audio,
            latency="high",     # permite ao driver usar buffers internos maiores
        )
        self.stream_audio.start()
        logger.info("Stream de áudio iniciado: %d Hz, blocksize=%d", TAXA_AUDIO, BLOCKSIZE_AUDIO)

        while self._rodando:
            # Bloqueia até 1 s; se a fila estiver vazia, volta ao topo do loop
            try:
                amostras = self._iq_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                if not (self.ouvindo_audio or self.gravando):
                    continue

                # 1. Decimação IQ (1.024 MHz → 256 kHz)
                iq_dec = amostras[::DECIMACAO_IQ]

                # 2. Filtro Butterworth passa-baixa — recalculado apenas quando a banda muda
                if self.banda_hz != banda_cache:
                    banda_cache = self.banda_hz
                    cutoff_norm = min(banda_cache / 2.0, _NYQUIST_POS_IQ - 1.0) / _NYQUIST_POS_IQ
                    b_band, a_band = signal.butter(ORDEM_FILTRO, cutoff_norm, btype="low")
                    zi_band = signal.lfilter_zi(b_band, a_band) * iq_dec[0]
                    logger.debug("Filtro recalculado: banda=%.0f Hz", self.banda_hz)

                iq_filt, zi_band = signal.lfilter(b_band, a_band, iq_dec, zi=zi_band)

                # 3. Desmodulação FM (diferença de fase entre amostras consecutivas)
                iq_completo = np.concatenate((prev_iq, iq_filt))
                prev_iq[0]  = iq_filt[-1]
                demodulado  = np.angle(iq_completo[1:] * np.conj(iq_completo[:-1]))

                # 4. Decimação de áudio (256 kHz → 32 kHz) + filtro de De-Emphasis (75 µs)
                audio_dec = demodulado[::DECIMACAO_AUDIO]
                audio_filt, zi_deemp = signal.lfilter(b_deemp, a_deemp, audio_dec, zi=zi_deemp)
                audio = (audio_filt * self.volume * 0.5).astype(np.float32)

                # 5. Escrita no ring buffer (SPSC)
                if self.ouvindo_audio:
                    self._escrever_ring(audio)

                # 6. Acumulação para IA e disparo de chunk
                if self.gravando:
                    self.buffer_ia.append(audio)
                    if len(self.buffer_ia) >= BLOCOS_POR_CHUNK:
                        fatia = np.concatenate(self.buffer_ia[:BLOCOS_POR_CHUNK])
                        self.buffer_ia = self.buffer_ia[BLOCOS_POR_CHUNK:]
                        if self.on_chunk_pronto:
                            self.on_chunk_pronto(fatia)
                        if self.on_verificar_termino:
                            self.on_verificar_termino()

            except Exception:
                logger.exception("Erro no DSP worker")

    def _escrever_ring(self, audio: np.ndarray) -> None:
        """Escrita circular no ring buffer (SPSC). Descarta amostras antigas em overflow."""
        n  = len(audio)
        wr = self.ring_write
        rd = self.ring_read
        espaco_livre = (rd - wr - 1) % TAMANHO_RING

        if n > espaco_livre:
            # Overflow: avança o ponteiro de leitura para abrir espaço
            excesso = n - espaco_livre
            self.ring_read = (rd + excesso) % TAMANHO_RING

        end = wr + n
        if end <= TAMANHO_RING:
            self.ring_buffer[wr:end] = audio
        else:
            parte1 = TAMANHO_RING - wr
            self.ring_buffer[wr:]      = audio[:parte1]
            self.ring_buffer[:n - parte1] = audio[parte1:]

        # Publica o novo ponteiro APÓS escrever os dados (garantia SPSC)
        self.ring_write = (wr + n) % TAMANHO_RING

    # =========================================================================
    # HELPERS DE ENCERRAMENTO
    # =========================================================================

    def _fechar_stream_audio(self) -> None:
        if self.stream_audio:
            try:
                self.stream_audio.stop()
                self.stream_audio.close()
            except Exception:
                logger.exception("Erro ao fechar stream de áudio")
            finally:
                self.stream_audio = None

    def _fechar_sdr(self) -> None:
        if self.sdr:
            try:
                self.sdr.close()
            except Exception:
                logger.exception("Erro ao fechar dispositivo RTL-SDR")
            finally:
                self.sdr = None
