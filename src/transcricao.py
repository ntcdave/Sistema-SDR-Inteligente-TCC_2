import whisper
import os
import csv
import sys
from datetime import datetime

# Garante que o diretório raiz do projeto está no path para importar config
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import CAMINHO_CSV, CAMINHO_BANCO  # noqa: E402

class TranscritorSDR:
    """
    Camada 2: A Escrita e o Banco de Dados.
    Responsável por converter o áudio em texto usando Whisper e salvar um histórico (CSV).
    """
    def __init__(self, modelo_tamanho="base"):
        print(f"🧠 Carregando o modelo Whisper ({modelo_tamanho})...")
        self.modelo = whisper.load_model(modelo_tamanho)

        # Usa o caminho centralizado do config.py (fonte única de verdade)
        os.makedirs(CAMINHO_BANCO, exist_ok=True)
        self.arquivo_csv = CAMINHO_CSV
        self._inicializar_csv()

    def _inicializar_csv(self):
        """Cria o cabeçalho do CSV se o arquivo ainda não existir."""
        if not os.path.exists(self.arquivo_csv):
            with open(self.arquivo_csv, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Data_Hora', 'Frequencia_MHz', 'Caminho_Audio', 'Texto_Transcrito'])

    def transcrever(self, caminho_audio, frequencia_mhz):
        """
        Transcreve o áudio e salva os dados no CSV.
        """
        if not os.path.exists(caminho_audio):
            print(f"❌ Erro: Arquivo de áudio não encontrado em {caminho_audio}")
            return ""

        print(f"📝 Transcrevendo áudio da rádio {frequencia_mhz} MHz...")
        try:
            resultado = self.modelo.transcribe(caminho_audio, fp16=False, language="pt")
            texto = resultado["text"].strip()
            
            # --- SALVANDO NO BANCO DE DADOS (CSV) ---
            if texto:
                data_atual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with open(self.arquivo_csv, mode='a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([data_atual, frequencia_mhz, caminho_audio, texto])
                print(f"💾 Transcrição salva no banco_transcricoes.csv com sucesso!")
            
            return texto
            
        except Exception as e:
            print(f"❌ Erro crítico na transcrição: {e}")
            return ""