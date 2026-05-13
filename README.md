# 📡 Sistema SDR Inteligente: Monitorização e Edge AI

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyQt6](https://img.shields.io/badge/PyQt-6-green.svg)
![Edge AI](https://img.shields.io/badge/Edge%20AI-Ollama%20%7C%20Whisper-orange)
![TCC](https://img.shields.io/badge/Projeto-TCC-purple.svg)

> Plataforma avançada de Inteligência de Sinais (SIGINT) que automatiza a captura, descodificação, transcrição e análise semântica de transmissões de rádio (FM) em tempo real, utilizando processamento 100% local (Edge Computing).

Este repositório contém o código-fonte desenvolvido para o Trabalho de Conclusão de Curso (TCC) em Engenharia. O foco principal é garantir privacidade, segurança e autonomia, processando dados sem qualquer dependência de serviços na nuvem.

*(Recomendamos a inclusão de uma imagem/screenshot da interface PyQt6 aqui)*

---

## ✨ Funcionalidades Principais

- **📻 Motor DSP Customizado:** Desmodulação de rádio FM em tempo real com numpy e scipy, aplicando decimação matemática e filtros Butterworth/De-Emphasis.
- **🎙️ Transcrição Offline (Speech-to-Text):** Conversão de áudio com ruídos em texto estruturado usando o modelo OpenAI Whisper (`base`).
- **🧠 Inteligência Analítica (LLM):** Classificação, extração de entidades e resumos usando Meta Llama 3.2 (1B) rodando localmente no Ollama.
- **📊 Dashboards Automáticos:** Geração de relatórios analíticos, gráficos de desempenho e linha do tempo de captações.
- **🖥️ Interface Gráfica Responsiva:** Desenvolvida em PyQt6 com visualização de espectro de radiofrequência em tempo real.

---

## 🛠️ Tecnologias Utilizadas

- **Interface:** `PyQt6`, `pyqtgraph`
- **DSP e Áudio:** `numpy`, `scipy`, `sounddevice`, hardware RTL-SDR Blog V4
- **Inteligência Artificial:** `openai-whisper` (PyTorch), `ollama` (Llama 3.2 1B)
- **Análise de Dados:** `pandas`, `matplotlib`, `seaborn`

---

## 🚀 Como Instalar e Executar

### Pré-requisitos Obrigatórios
1. **Python 3.10 ou superior:** Instalado no sistema.
2. **FFmpeg:** Obrigatório para o Whisper. Instale e garanta que está [adicionado ao PATH do Windows](https://phoenixnap.com/kb/ffmpeg-windows).
3. **Antena RTL-SDR:** Conectada via USB. Você deve instalar os drivers WinUSB corretos usando o software [Zadig](https://zadig.akeo.ie/).
4. **Ollama:** Instale o [Ollama](https://ollama.com/) e deixe o servidor rodando em segundo plano.

### Passo a Passo

**1. Clonar o Repositório:**
```bash
git clone https://github.com/ntcdave/Sistema-SDR-Inteligente-TCC_2.git
cd Sistema-SDR-Inteligente-TCC_2
```

**2. Instalar Dependências:**
Recomendamos o uso de um ambiente virtual (venv).
```bash
pip install -r requirements.txt
```

**3. Descarregar os Modelos de IA:**
O modelo Whisper (`base`) será baixado automaticamente na primeira execução (~140MB). Para a análise semântica, baixe o modelo Llama via Ollama:
```bash
ollama pull llama3.2:1b
```

**4. Iniciar a Aplicação:**
Com a antena ligada e configurada, execute:
```bash
python app.py
```

---

## 📂 Organização do Projeto

```text
projeto/
├── app.py                  # Interface principal (PyQt6) e motor DSP
├── requirements.txt        # Lista de dependências Python
├── src/
│   ├── transcricao.py      # Interação com o OpenAI Whisper (STT)
│   ├── analise.py          # Interrogador do Llama 3.2 e gerador de gráficos
│   └── captura.py          # Módulo legado de processamento
├── dados/                  # Banco de dados CSVs, relatórios e chunks (.wav)
└── ferramentas/rtl-sdr/    # DLLs e utilitários da RTL-SDR para Windows
```

---

## 📖 Documentação Técnica Avançada

Para entender a fundo a arquitetura do projeto, fluxo de dados do Processamento Digital de Sinal (DSP), padrões de projeto aplicados e pontos de extensibilidade, consulte a nossa documentação completa:

👉 **[Ler a Especificação Completa (SPEC.me)](./SPEC.me)**

---

## 🤝 Contribuição e Resolução de Problemas (Troubleshooting)

Este projeto é **Open Source**! A comunidade é muito bem-vinda para explorar o código, abrir Issues para reportar problemas ou sugerir melhorias através de Pull Requests. 

Caso encontre alguma dificuldade inicial na execução, confira as soluções para os problemas mais comuns relatados pela comunidade:

- **Erro na biblioteca RTL-SDR:** Verifique se a pasta `ferramentas/rtl-sdr/` contém a `rtlsdr.dll` e se você configurou os drivers corretos usando o Zadig. (Usuários Linux/macOS podem precisar compilar a biblioteca localmente).
- **Falha ao Transcrever Áudio:** Se a aplicação apresentar erros durante o Whisper, o `ffmpeg` provavelmente não foi encontrado pelo sistema. Assegure-se de que ele está instalado e configurado no PATH.
- **Análise Semântica não inicia:** A IA necessita do servidor Ollama ativo. Verifique se o `ollama serve` está sendo executado em segundo plano.

Sentiu falta de alguma funcionalidade ou conseguiu resolver um bug diferente? **Contribua com o projeto abrindo uma Pull Request!**

---

> *Trabalho académico desenvolvido para investigar a implementação de técnicas de Edge AI aplicadas a Radiofrequência e processamento DSP em tempo real.*
