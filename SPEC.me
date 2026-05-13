# 📡 SPEC.me — Bíblia Técnica do Sistema SDR Inteligente

> **Projeto:** Sistema SDR Inteligente — Monitorização e Edge AI  
> **Tipo:** Trabalho de Conclusão de Curso (TCC) em Engenharia  
> **Gerado em:** Maio de 2026  
> **Versão do Documento:** 1.0

---

## Sumário

1. [Arquitetura do Sistema](#1-arquitetura-do-sistema)
2. [Stack Técnica](#2-stack-técnica)
3. [Dicionário de Funcionalidades](#3-dicionário-de-funcionalidades)
4. [Padrões de Projeto](#4-padrões-de-projeto)
5. [Pontos de Extensibilidade](#5-pontos-de-extensibilidade)
6. [Guia de Manutenção](#6-guia-de-manutenção)

---

## 1. Arquitetura do Sistema

### 1.1 Visão Geral

O sistema opera como um **pipeline de três camadas** que transformam ondas eletromagnéticas em inteligência acionável, executando tudo localmente (Edge Computing):

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  ANTENA      │───▸│  MOTOR DSP       │───▸│  WHISPER (STT)   │───▸│  LLAMA 3.2      │
│  RTL-SDR V4  │    │  (app.py)        │    │  (transcricao.py)│    │  (analise.py)   │
│  Hardware    │    │  Desmodulação FM  │    │  Áudio → Texto   │    │  Texto → Intel  │
└─────────────┘    └──────────────────┘    └──────────────────┘    └─────────────────┘
       │                    │                       │                       │
       │              ┌─────▼──────┐          ┌─────▼──────┐         ┌─────▼──────┐
       │              │ sounddevice│          │  CSV Bank  │         │ Dashboard  │
       │              │ (Áudio ao  │          │ (Memória   │         │ (matplotlib│
       │              │  vivo)     │          │ persistente│         │ + seaborn) │
       └──────────────┴────────────┘          └────────────┘         └────────────┘
```

### 1.2 Fluxo de Dados Detalhado

**Fase 1 — Captura e Desmodulação (Camada DSP em `app.py`)**

1. O `RtlSdr` abre o dispositivo USB e configura sample rate (1.024 MHz), frequência central e ganho.
2. O loop `thread_master_loop` lê blocos de **102.400 amostras I/Q** continuamente.
3. As amostras passam por uma **cadeia de DSP matemático puro**:
   - **Decimação I/Q (÷4):** Reduz a taxa de 1.024 MHz para 256 kHz via média aritmética por blocos.
   - **Filtro Butterworth passa-baixa (ordem 3):** Isola a banda do canal FM sintonizado. O filtro é recalculado apenas quando `banda_atual` muda (cache inteligente).
   - **Desmodulação FM:** Calcula a diferença de fase entre amostras consecutivas via `np.angle(z[n] * conj(z[n-1]))`.
   - **Decimação de áudio (÷8):** Reduz de 256 kHz para **32 kHz** (taxa de reprodução final).
   - **Filtro De-Emphasis (75µs):** Remove a pré-ênfase padrão das emissões FM, suavizando o áudio.
4. O áudio final é multiplicado pelo volume e injectado no `buffer_audio` (thread-safe via `threading.Lock`).

**Fase 2 — Transcrição (Camada STT em `src/transcricao.py`)**

5. Quando a gravação IA está ativa, blocos de áudio são acumulados no `buffer_ia`.
6. A cada **~300 blocos (~30 segundos)**, o sistema salva um ficheiro `chunk.wav` (PCM 16-bit, 32 kHz, mono).
7. O `TranscritorSDR` carrega o modelo Whisper (`base`) e transcreve o áudio forçando idioma português (`language="pt"`).
8. O texto resultante é persistido num ficheiro CSV com timestamp, frequência e caminho do áudio.

**Fase 3 — Análise Semântica (Camada IA em `src/analise.py`)**

9. O `CientistaSDR` lê as últimas N transcrições do CSV.
10. Para cada transcrição, executa em paralelo:
    - **Análise Quantitativa:** Tokenização, remoção de stopwords PT-BR, contagem de frequência (Counter).
    - **Análise Qualitativa (LLM):** Envia prompt estruturado ao Llama 3.2:1b via Ollama, pedindo categorização, extração de entidades e resumo.
11. Ambas as análises são instrumentadas com `tracemalloc` para medição de pico de RAM.
12. São gerados **5 gráficos científicos** (PNG 300dpi) e um relatório CSV consolidado.

### 1.3 Módulo Legado — `src/captura.py`

O `SDRReceiver` é uma implementação **anterior** baseada em subprocessos (`rtl_fm.exe` → `sox.exe` via pipe). Foi substituída pelo motor DSP nativo em `app.py`, mas permanece no codebase como referência arquitetural. Ele não é importado por nenhum módulo ativo.

---

## 2. Stack Técnica

### 2.1 Interface Gráfica

| Biblioteca | Versão | Justificativa |
|---|---|---|
| **PyQt6** | — | Framework de GUI desktop com suporte nativo a widgets, layouts responsivos e `QTimer` para integração thread-safe com a UI. Escolhido por maturidade e capacidade de construir interfaces complexas sem servidor web. |
| **pyqtgraph** | — | Renderização de espectrogramas em tempo real com performance GPU-acelerada. Superior ao matplotlib para dados dinâmicos (atualização a cada 50ms). |

### 2.2 Processamento Digital de Sinal

| Biblioteca | Justificativa |
|---|---|
| **numpy** | Operações vetorizadas sobre arrays I/Q (reshape, mean, angle, conj). Essencial para processar 102.400 amostras por ciclo sem degradação de performance. |
| **scipy.signal** | Implementação dos filtros Butterworth (`butter`) e filtragem stateful (`lfilter` com `zi`). O estado `zi` é preservado entre iterações para evitar transientes nas fronteiras de blocos. |
| **sounddevice** | Stream de áudio em tempo real via callback. O `OutputStream` com `blocksize=0` permite latência adaptativa, evitando stutters. |

### 2.3 Inteligência Artificial

| Componente | Modelo | Justificativa |
|---|---|---|
| **openai-whisper** | `base` (~140MB) | Modelo de STT robusto contra ruído de rádio. A variante `base` equilibra precisão e velocidade em hardware consumer. O `fp16=False` garante compatibilidade com CPUs sem suporte a half-precision. |
| **torch** | — | Backend de inferência do Whisper. Dependência obrigatória. |
| **ollama** | Llama 3.2:1b (~1.3GB) | LLM compacto executado localmente via servidor Ollama. A escolha do modelo 1B permite inferência em máquinas sem GPU dedicada, com `num_ctx=1024` e `temperature=0.1` para respostas determinísticas e concisas. |

### 2.4 Análise e Visualização

| Biblioteca | Justificativa |
|---|---|
| **pandas** | Leitura e manipulação do CSV de transcrições. Operações como `tail()`, `groupby()` e conversão temporal (`to_datetime`). |
| **matplotlib** (backend `Agg`) | Geração de gráficos estáticos em alta resolução (300 dpi) para inclusão em documentos acadêmicos. O backend `Agg` é não-interativo, evitando conflitos com a thread da GUI PyQt6. |
| **seaborn** | Camada estética sobre matplotlib. Paletas científicas (`magma`, `viridis`, `coolwarm`, `Set2`, `crest`) e tema `whitegrid` para publicação acadêmica. |

### 2.5 Hardware e Drivers

| Componente | Descrição |
|---|---|
| **pyrtlsdr** | Binding Python para `librtlsdr`. Permite controle programático da antena (frequência, ganho, sample rate). |
| **rtlsdr.dll** + utilitários | DLLs e executáveis nativos Windows distribuídos em `ferramentas/rtl-sdr/`. O PATH é modificado em runtime para garantir linkagem dinâmica. |

---

## 3. Dicionário de Funcionalidades

### 3.1 `app.py` — Orquestrador Principal

#### Classe `MainWindow(QMainWindow)`

| Método | Responsabilidade |
|---|---|
| `__init__` | Inicializa estado (frequência, ganho, volume, buffers), cria `TranscritorSDR`, constrói a interface e dispara o hardware em background. |
| `closeEvent` | Shutdown gracioso: sinaliza `hardware_rodando=False`, espera 300ms, fecha stream de áudio e libera o dispositivo SDR. |
| `construir_interface` | Monta toda a UI: painel lateral com scroll (sliders, botões, frames de captura e análise, log de texto) e gráfico de espectro bloqueado para interação do rato. |
| `escolher_pasta` | Abre diálogo nativo para o utilizador redefinir o diretório de destino dos chunks `.wav`. |
| `escolher_csv` | Abre diálogo para selecionar um CSV alternativo como fonte de dados para a análise semântica. |
| `centralizar_grafico` | Recalcula o range X do espectrograma com base na frequência sintonizada e no nível de zoom. Atualiza a região visual da banda (faixa vermelha). |
| `mudar_frequencia` | Converte o valor do slider (inteiro ×10) para MHz, atualiza a label, reconfigura o `center_freq` do SDR e recentra o gráfico. |
| `mudar_ganho` | **Algoritmo de Ganho Inteligente:** consulta `valid_gains_db` do hardware, encontra o degrau discreto mais próximo do valor desejado, e aplica-o. Garante que o valor exibido na UI corresponde ao ganho real da antena. |
| `mudar_volume` | Escala linear de volume (0–200% → 0.0–2.0). |
| `mudar_zoom` / `mudar_altura_grafico` | Controlam os eixos X e Y do espectrograma respectivamente. |
| `toggle_audio` | Liga/desliga a reprodução ao vivo. Ao ligar, limpa o buffer para evitar eco de áudio antigo. |
| `callback_audio` | Callback do `sounddevice`: alimenta o output stream com amostras do buffer thread-safe. Se o buffer estiver vazio, envia silêncio (zeros). |
| `iniciar_hardware_background` | Inicia o `QTimer` de atualização gráfica (50ms) e dispara `thread_master_loop` como daemon thread. |
| `thread_master_loop` | **Loop principal do DSP.** Configura o SDR, cria o stream de áudio, e executa o ciclo infinito de leitura → desmodulação → reprodução/gravação. Contém toda a cadeia de filtros com estado persistente entre iterações. |
| `atualizar_grafico` | Calcula a PSD (Power Spectral Density) via FFT sobre 4096 amostras, converte para dB, e atualiza a curva do pyqtgraph. |
| `toggle_missao` | State machine da captura: valida parâmetros do modo selecionado (contínuo, tempo fixo, até horário), inicializa buffers e altera estado visual do botão. |
| `verificar_termino` | Verifica condições de paragem automática (duração atingida ou horário-alvo alcançado). Dispara `toggle_missao` via `QTimer.singleShot` para operar na thread da UI. |
| `processar_chunk` | Salva buffer de áudio como WAV (PCM 16-bit, 32kHz, mono), invoca o transcritor Whisper, e atualiza o log da interface. Executado em thread separada. |
| `abrir_analise` | Desabilita o botão e dispara `rodar_analise` em background. |
| `rodar_analise` | Importa `CientistaSDR` sob demanda (lazy import), instancia com o CSV selecionado, e executa a análise com limite configurável. |

### 3.2 `src/transcricao.py` — Camada de Escrita

#### Classe `TranscritorSDR`

| Método | Responsabilidade |
|---|---|
| `__init__(modelo_tamanho)` | Carrega o modelo Whisper na RAM, garante existência da pasta `dados/`, e inicializa o CSV. |
| `_inicializar_csv` | Cria o ficheiro CSV com cabeçalho (`Data_Hora`, `Frequencia_MHz`, `Caminho_Audio`, `Texto_Transcrito`) apenas se ele ainda não existir. Idempotente. |
| `transcrever(caminho_audio, frequencia_mhz)` | Valida existência do ficheiro, executa `model.transcribe()` com `fp16=False` e `language="pt"`, e persiste o resultado no CSV via append. Retorna o texto ou string vazia em caso de erro. |

### 3.3 `src/analise.py` — Motor Analítico

#### Classe `CientistaSDR`

| Método | Responsabilidade |
|---|---|
| `__init__(caminho_csv)` | Configura o modelo LLM (`llama3.2:1b`), define o CSV fonte (parametrizável ou padrão), e inicializa listas de acumulação para métricas e dados agregados. |
| `limpar_texto(texto)` | Normaliza texto: converte para minúsculas e remove toda pontuação via regex `[^\w\s]`. |
| `analise_quantitativa(texto)` | Pipeline estatístico: limpeza → tokenização → remoção de **70+ stopwords** PT-BR/PT-PT → contagem de frequência → top 5. Instrumentado com `tracemalloc` para benchmarking de memória. |
| `analise_qualitativa_llm(texto)` | Constrói prompt estruturado e envia ao Ollama. Extrai a categoria via regex e acumula no histórico. Parâmetros LLM: contexto de 1024 tokens, máximo 150 tokens de predição, temperatura 0.1. |
| `gerar_graficos_separados(df)` | Gera 5 figuras científicas: (1) barras de termos frequentes, (2) pizza de categorias, (3) barras de entidades com word-wrap, (4) subplot duplo de desempenho CPU vs LLM (tempo + RAM), (5) série temporal de interceptações por hora. Todas salvas em 300 dpi. |
| `executar_analise(limite)` | Orquestrador: lê CSV → seleciona últimos N registros → executa análise dual por registro → gera gráficos → exporta relatório CSV consolidado (`relatorio_alertas_tcc.csv`). |

### 3.4 `src/captura.py` — Módulo Legado

#### Classe `SDRReceiver`

| Método | Responsabilidade |
|---|---|
| `__init__(...)` | Configura parâmetros de captura (frequência, duração, caminhos de `rtl_fm.exe` e `sox.exe`). |
| `record_audio` | Cria pipeline de subprocessos `rtl_fm → sox` via pipe stdout/stdin. Gerencia ciclo de vida dos processos com terminação explícita e tempo de recuperação USB. **Não utilizado ativamente.** |

---

## 4. Padrões de Projeto

### 4.1 Producer-Consumer (com Buffer Thread-Safe)

O padrão mais crítico do sistema. A `thread_master_loop` (producer) gera amostras de áudio processado e as deposita no `buffer_audio`. O `callback_audio` do sounddevice (consumer) consome essas amostras em ritmo determinado pelo driver de áudio. A sincronização é garantida por `threading.Lock`, e o buffer possui limite de 2 segundos para evitar acúmulo de latência.

### 4.2 Observer / Event-Driven (Qt Signals & Slots)

Todos os sliders e botões da interface utilizam o sistema de signals/slots do Qt: `slider.valueChanged.connect(self.handler)`. Isso desacopla a geração de eventos da lógica de negócio, permitindo que a UI reaja a mudanças de estado sem polling.

### 4.3 Lazy Import

O módulo `src/analise.py` (e sua pesada cadeia de dependências: pandas, matplotlib, seaborn, ollama) é importado **apenas quando o utilizador clica em "Gerar Dashboard"** (dentro de `rodar_analise`). Isso evita carregar ~200MB+ de bibliotecas no arranque da aplicação.

### 4.4 Template Method (Implícito)

O `executar_analise` funciona como um template method: define a sequência fixa (ler CSV → iterar registos → análise quantitativa → análise qualitativa → gerar gráficos → exportar), delegando cada passo a métodos especializados.

### 4.5 Stateful Filter Chain

Os filtros DSP (Butterworth e De-Emphasis) mantêm estado (`zi`) entre iterações do loop. Este padrão garante continuidade do sinal entre blocos de 102.400 amostras, eliminando artefatos de fronteira (cliques e descontinuidades).

### 4.6 Graceful Shutdown

O `closeEvent` implementa um protocolo de encerramento ordenado: sinaliza → espera drenagem → fecha streams → libera hardware. Previne corrupção de estado do dispositivo USB.

### 4.7 Smart Snapping (Quantização de Ganho)

O `mudar_ganho` implementa um padrão de adaptação: o valor contínuo do slider é mapeado para o valor discreto mais próximo aceite pelo hardware (`min(..., key=lambda)`). A UI é atualizada com o valor real, não o desejado.

---

## 5. Pontos de Extensibilidade

### 5.1 Multi-Frequência e Scanning

O sistema já possui separação entre `frequencia_atual` e o loop de captura. É possível implementar um **scanner automático** que varra uma faixa de frequências, sintonizando cada uma por N segundos e acumulando transcrições. O `toggle_missao` já suporta modos de captura parametrizáveis via `combo_modo`.

### 5.2 Modelos de IA Intercambiáveis

- **Whisper:** O `modelo_tamanho` é parametrizado no construtor de `TranscritorSDR`. Trocar de `base` para `small`, `medium` ou `large` requer apenas alterar essa string.
- **LLM:** O `modelo_llm` em `CientistaSDR` é uma string (`llama3.2:1b`). Qualquer modelo compatível com Ollama pode ser usado sem alterar código (ex: `mistral`, `gemma`, `phi3`).

### 5.3 Persistência e Banco de Dados

O CSV é o mecanismo atual de persistência. A interface `escolher_csv` já permite operar sobre ficheiros distintos. A migração para SQLite ou PostgreSQL requer apenas substituir os métodos `_inicializar_csv` e o bloco de escrita em `transcrever`, mantendo a mesma interface pública.

### 5.4 Novos Tipos de Análise

O `executar_analise` itera sobre registros e delega a métodos individuais. Novas análises (ex: análise de sentimento, detecção de alertas urgentes, correlação cruzada entre frequências) podem ser adicionadas como novos métodos na classe `CientistaSDR` e chamadas dentro do loop de iteração.

### 5.5 Exportação de Relatórios

O sistema já gera PNGs e CSVs. A adição de exportação para PDF (via `reportlab` ou `fpdf2`), HTML interativo (via `plotly`), ou integração com APIs externas (Telegram, webhook) pode ser feita como módulos adicionais na pasta `src/`.

### 5.6 Modos de Modulação

O motor DSP em `thread_master_loop` implementa desmodulação WBFM. O pipeline de filtros é modular: substituir o bloco de desmodulação (linha 353 de `app.py`) por outros esquemas (AM, SSB, NBFM) é viável sem refatoração estrutural. O `SDRReceiver` legado já referencia o parâmetro `-M wbfm`, indicando que a arquitetura contemplava múltiplos modos.

### 5.7 Dashboard em Tempo Real

O `pyqtgraph` já renderiza o espectro em tempo real. É possível adicionar novos widgets (waterfall/spectrogram 2D, medidor de SNR, indicador de nível de áudio) como novos `PlotWidget` no layout principal.

---

## 6. Guia de Manutenção

### 6.1 Dependências Críticas e Versões

| Dependência | Risco | Observação |
|---|---|---|
| **PyQt6** | 🟡 Médio | Incompatível com PyQt5. Migrar para PySide6 requer renomear imports (API idêntica). |
| **openai-whisper** | 🔴 Alto | Depende de `torch` e `ffmpeg`. O `ffmpeg` **deve estar no PATH** do sistema para a transcrição funcionar. Não está listado no `requirements.txt`. |
| **pyrtlsdr** | 🟡 Médio | Depende da `rtlsdr.dll` presente em `ferramentas/rtl-sdr/`. O PATH é modificado em runtime (linha 15 de `app.py`). Se a DLL não for encontrada, o import falha silenciosamente até a primeira chamada `RtlSdr()`. |
| **ollama** | 🟡 Médio | Requer o servidor Ollama rodando em `localhost:11434`. Se o servidor estiver offline, `analise_qualitativa_llm` captura a exceção mas classifica como "Erro". |

### 6.2 Configuração de Ambiente

1. **Drivers USB:** A antena RTL-SDR requer drivers WinUSB instalados via **Zadig**. Sem eles, `RtlSdr()` lançará exceção.
2. **Servidor Ollama:** Deve estar rodando antes de clicar em "Gerar Dashboard". Instalar e executar: `ollama serve` + `ollama pull llama3.2:1b`.
3. **ffmpeg:** Dependência implícita do Whisper. Instalar via `choco install ffmpeg` ou adicionar manualmente ao PATH.
4. **Python 3.10+:** Necessário para compatibilidade com PyQt6 e tipagem moderna.

### 6.3 Ficheiros Não Versionados (`.gitignore`)

- `dados/brutos/` e `*.wav` — Chunks de áudio bruto (podem atingir GBs).
- `dados/graficos_tcc/` — PNGs gerados pela análise.
- `venv/`, `__pycache__/` — Ambientes virtuais e bytecode.

### 6.4 Pontos de Atenção no Código

| Local | Observação |
|---|---|
| `app.py:73-78` | `except: pass` genérico no `closeEvent`. Pode ocultar erros de liberação de hardware. Recomenda-se logging. |
| `app.py:308` | `except: pass` no arranque de ganho. Falhas de comunicação com a antena são silenciadas. |
| `app.py:372-373` | `except: pass` no loop principal DSP. Erros de leitura do SDR (USB desconectado) são ignorados, o loop continua tentando. |
| `app.py:367` | O threshold de 300 blocos para chunk é um **magic number**. Deveria ser uma constante nomeada. |
| `config.py` | Ficheiro **vazio**. Indica intenção de centralizar configurações que atualmente estão hardcoded em `app.py` (frequência padrão, ganho, taxa de amostragem, etc.). |
| `main.py` | Ficheiro **vazio**. O entry point real é `app.py`. Pode indicar plano futuro de separação CLI vs GUI. |
| `src/captura.py` | Módulo **órfão** — não é importado por nenhum ficheiro ativo. Referencia `sox.exe` que não está incluído em `ferramentas/`. |
| `analise.py:226` | `os.startfile()` é **Windows-only**. Quebra em Linux/macOS. |
| `transcricao.py:20-21` | O CSV de transcrições é criado em `dados/` (raiz), mas `app.py` espera-o em `dados/banco_dados/`. Isso cria **dois ficheiros CSV distintos** se ambos os caminhos forem utilizados. |

### 6.5 Performance e Limites

- **Buffer de áudio:** Limitado a 2 segundos (`taxa_audio * 2`). Se o processamento atrasar, áudio antigo é descartado silenciosamente.
- **Memória Whisper:** O modelo `base` ocupa ~140MB de RAM permanentemente após carregamento no `__init__` do `TranscritorSDR`.
- **Chunks concorrentes:** Cada chunk de 30s dispara uma nova thread (`processar_chunk`). Não há pool ou limite de threads. Capturas longas podem gerar dezenas de threads simultâneas.
- **Gráficos matplotlib:** O backend `Agg` e `gc.collect()` explícito mitigam vazamentos de memória, mas sessões muito longas de análise podem acumular objetos temporários.

### 6.6 Estrutura de Diretórios em Runtime

```
projeto/
├── app.py                          # Entry point e motor DSP
├── config.py                       # (Vazio — reservado)
├── main.py                         # (Vazio — reservado)
├── requirements.txt                # Dependências pip
├── src/
│   ├── transcricao.py              # Whisper STT + persistência CSV
│   ├── analise.py                  # Llama 3.2 + dashboards
│   └── captura.py                  # (Legado — não utilizado)
├── dados/
│   ├── banco_transcricoes.csv      # CSV gerado por transcricao.py
│   ├── banco_dados/                # Diretório alternativo de CSV (app.py)
│   ├── brutos/                     # Chunks WAV organizados por data/hora
│   │   └── 2026-05-13/
│   │       └── 15h30m00s/
│   │           └── chunk.wav
│   └── graficos_tcc/               # PNGs gerados pela análise
│       ├── *_fig1_palavras.png
│       ├── *_fig2_categorias.png
│       ├── *_fig3_entidades.png
│       ├── *_fig4_desempenho.png
│       └── *_fig5_linha_do_tempo.png
└── ferramentas/
    └── rtl-sdr/                    # DLLs e binários nativos Windows
        ├── rtlsdr.dll
        ├── rtl_fm.exe
        └── ...
```

---

> **Nota Final:** Este documento reflete o estado do codebase na data de geração. Alterações subsequentes no código devem ser refletidas com atualizações neste ficheiro para manter sua utilidade como referência canónica do sistema.
