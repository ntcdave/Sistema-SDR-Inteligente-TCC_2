📡 Sistema SDR Inteligente: Monitorização e Edge AI

Este repositório contém o código-fonte desenvolvido para o Trabalho de Conclusão de Curso (TCC) em Engenharia. O projeto consiste numa plataforma avançada de Inteligência de Sinais (SIGINT) que automatiza a captura, descodificação, transcrição e análise semântica de transmissões de rádio (FM) em tempo real.

Todo o processamento é realizado localmente (Edge Computing), garantindo privacidade, segurança e autonomia face a serviços baseados na Cloud.

🛠️ Arquitetura do Sistema e Tecnologias

A aplicação foi estruturada em três camadas principais: Processamento Digital de Sinal (DSP), Reconhecimento de Voz e Inteligência Artificial Semântica.

1. Hardware e Motor DSP (Processamento de Sinal)

Hardware: Antena RTL-SDR Blog V4 (Chipset R828D). O repositório já inclui os drivers e bibliotecas estáticas (.dll e .lib) necessárias para comunicação nativa em Windows na pasta ferramentas/rtl-sdr/.

Motor DSP (Python): Utiliza numpy e scipy para realizar a decimação I/Q, filtragem passa-baixa (Butterworth), desmodulação FM matemática e aplicação de filtros De-Emphasis.

Buffer de Áudio: O áudio desmodulado é gerido pela biblioteca sounddevice, utilizando uma latência preventiva que impede estrangulamentos (stutters) na interface.

2. Transcrição Automática (Speech-to-Text)

A plataforma guarda o áudio da rádio em blocos contínuos de 30 segundos (chunks). Estes ficheiros .wav são passados para o primeiro modelo de Inteligência Artificial:

Modelo Utilizado: OpenAI Whisper (Versão: base)

O que é descarregado? Ao executar a transcrição pela primeira vez, o código descarrega automaticamente os pesos do modelo (base.pt, com cerca de ~140MB) para a cache da sua máquina.

Função: Converte o áudio com estática e ruído de rádio em texto limpo e estruturado.

3. Análise Semântica (Edge LLM)

Os textos transcritos são analisados num painel analítico alimentado por um Large Language Model otimizado para hardware limitado:

Modelo Utilizado: Meta Llama 3.2 (Versão: 1B) via servidor local Ollama.

O que é descarregado? Requer a instalação do Ollama e a execução do comando ollama pull llama3.2:1b (descarrega o modelo comprimido de aproximadamente ~1.3GB).

Função: O Llama 3.2 lê o texto transcrito, avalia o contexto, classifica a transmissão (ex: Música, Jornalismo, Trânsito), elabora um resumo tático e extrai entidades cruciais (Locais, Pessoas e Marcas).

🚀 Como Instalar e Executar

Pré-requisitos

Python 3.10 ou superior: Instalado no sistema.

Antena RTL-SDR: Conectada via USB e com os drivers WinUSB instalados através do software Zadig.

Servidor Ollama: Instale o Ollama e deixe-o a correr em segundo plano.

Passo a Passo

Clonar o Repositório:

git clone [https://github.com/ThayBellona/projeto_tcc_sdr.git](https://github.com/ThayBellona/projeto_tcc_sdr.git)
cd projeto_tcc_sdr


Instalar Dependências:
Instale todas as bibliotecas necessárias listadas no requirements.txt:

pip install -r requirements.txt


Descarregar o Modelo de Análise (Llama 3.2):

ollama pull llama3.2:1b


Iniciar a Interface Gráfica:
Com a antena ligada e configurada, execute a aplicação principal:

python app.py


📂 Organização do Repositório

app.py: Interface principal desenvolvida em PyQt6. Gere a linha de montagem: antena, áudio ao vivo e captura de chunks.

src/transcricao.py: Classe responsável por interagir com a API local do Whisper.

src/analise.py: Motor analítico. Interroga o Llama 3.2, mede picos de Memória RAM e tempos de processamento (tracemalloc), e desenha dashboards complexos usando matplotlib e seaborn.

dados/banco_dados/: Onde são gerados os ficheiros .csv que funcionam como a memória persistente das captações de rádio.

ferramentas/rtl-sdr/: Condensador de ferramentas auxiliares e bibliotecas de acesso direto ao SDR.

Trabalho académico desenvolvido para investigar a implementação de técnicas de Edge AI aplicadas a Radiofrequência e processamento DSP em tempo real.
