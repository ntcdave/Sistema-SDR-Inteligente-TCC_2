import whisper
import os
import csv
from datetime import datetime

class TranscritorSDR:
    """
    Camada 2: A Escrita e o Banco de Dados.
    Responsável por converter o áudio em texto usando Whisper e salvar um histórico (CSV).
    """
    def __init__(self, modelo_tamanho="base"):
        print(f"🧠 Carregando o modelo Whisper ({modelo_tamanho})...")
        self.modelo = whisper.load_model(modelo_tamanho)
        
        # Garante que a pasta de dados existe
        self.pasta_dados = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dados'))
        os.makedirs(self.pasta_dados, exist_ok=True)
        
        # Cria a base de dados CSV
        self.arquivo_csv = os.path.join(self.pasta_dados, 'banco_transcricoes.csv')
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