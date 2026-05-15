"""
config.py — Configurações centralizadas do Sistema SDR Inteligente.

Todas as constantes do projeto residem aqui.
Nenhum valor hardcoded deve existir em app.py, src/dsp.py ou src/transcricao.py.
"""

import os

# =============================================================================
# CAMINHOS
# =============================================================================

BASE_DIR      = os.path.abspath(os.path.dirname(__file__))
CAMINHO_DLL   = os.path.join(BASE_DIR, "ferramentas", "rtl-sdr")
CAMINHO_DADOS = os.path.join(BASE_DIR, "dados")
CAMINHO_BRUTOS = os.path.join(CAMINHO_DADOS, "brutos")
CAMINHO_BANCO  = os.path.join(CAMINHO_DADOS, "banco_dados")
CAMINHO_CSV    = os.path.join(CAMINHO_BANCO, "banco_transcricoes.csv")  # fonte única de verdade

# =============================================================================
# HARDWARE RTL-SDR
# =============================================================================

FREQUENCIA_PADRAO_MHZ: float = 100.9
GANHO_PADRAO_DB: float       = 40.0
SAMPLE_RATE_SDR: int         = 1_024_000   # Hz — taxa de amostragem do RTL-SDR
TAMANHO_BLOCO_SDR: int       = 262_144     # amostras por leitura USB (~250 ms);
                                            # deve ser múltiplo de 16 384 (exigência do driver)

# =============================================================================
# PIPELINE DSP
# =============================================================================

DECIMACAO_IQ: int    = 4     # 1.024 MHz → 256 kHz
DECIMACAO_AUDIO: int = 8     # 256 kHz   →  32 kHz
TAXA_AUDIO: int      = SAMPLE_RATE_SDR // DECIMACAO_IQ // DECIMACAO_AUDIO  # 32 000 Hz

DEEMPHASIS_TAU: float  = 75e-6     # constante de tempo do filtro de de-emphasis (padrão FM)
BANDA_PADRAO_HZ: float = 170_000.0 # largura de banda do canal FM (WBFM)
ORDEM_FILTRO: int      = 3         # ordem do filtro Butterworth passa-baixa

# =============================================================================
# RING BUFFER DE ÁUDIO (SPSC, sem lock)
# =============================================================================

TAMANHO_RING: int  = TAXA_AUDIO * 4  # 4 s @ 32 kHz = 128 000 amostras
BLOCKSIZE_AUDIO: int = 2_048          # frames por callback do sounddevice (~64 ms @ 32 kHz)

# =============================================================================
# CAPTURA E TRANSCRIÇÃO
# =============================================================================

BLOCOS_POR_CHUNK: int = 300       # nº de blocos DSP antes de salvar um chunk .wav (~30 s)
MODELO_WHISPER: str   = "base"    # "base" | "small" | "medium" | "large"
MAX_WORKERS_WHISPER: int = 2      # threads paralelas de transcrição Whisper

# =============================================================================
# ANÁLISE SEMÂNTICA (LLM)
# =============================================================================

MODELO_LLM: str      = "llama3.2:1b"
LLM_NUM_CTX: int     = 1_024
LLM_NUM_PREDICT: int = 150
LLM_TEMPERATURE: float = 0.1

# =============================================================================
# INTERFACE GRÁFICA
# =============================================================================

INTERVALO_TIMER_MS: int      = 120    # período do QTimer do espectrograma (~8 FPS)
FFT_SIZE: int                = 4_096  # amostras para cálculo da PSD
LARGURA_PAINEL_LATERAL: int  = 380    # px — largura fixa do painel de controlos
LIMITE_ANALISE_REGISTOS: int = 5      # nº de transcrições enviadas ao LLM por sessão
