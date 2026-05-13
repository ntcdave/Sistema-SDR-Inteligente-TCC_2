import subprocess
import os
import time

class SDRReceiver:
    """
    Camada 1: O Ouvido (Versão com Gerenciamento de Memória e Hardware).
    """
    def __init__(self, frequency="104.9M", duration=30, output_dir="dados/brutos", filename="gravacao_atual.wav"):
        self.frequency = frequency
        self.duration = duration
        self.output_file = os.path.join(output_dir, filename)
        
        self.rf_gain = "40"      
        self.bandwidth = "170k"  
        self.audio_rate = "48k"  
        
        self.caminho_rtl_fm = "ferramentas/rtl-sdr/rtl_fm.exe"
        self.caminho_sox = "ferramentas/sox/sox.exe"
        
    def record_audio(self):
        print(f"📻 Sintonizando {self.frequency} | Ganho manual: {self.rf_gain}...")
        
        command_rtl = [
            self.caminho_rtl_fm, 
            "-M", "wbfm",           
            "-f", self.frequency,   
            "-s", self.bandwidth,   
            "-r", self.audio_rate,  
            "-g", self.rf_gain, 
            "-E", "deemp",          
            "-"                     
        ]
        
        command_sox = [
            self.caminho_sox, 
            "-t", "raw", "-r", self.audio_rate, "-e", "signed", "-b", "16", "-c", "1", "-", 
            self.output_file, 
            "trim", "0", str(self.duration) 
        ]

        try:
            # 1. Inicia os dois processos
            process_rtl = subprocess.Popen(command_rtl, stdout=subprocess.PIPE, stderr=None)
            process_sox = subprocess.Popen(command_sox, stdin=process_rtl.stdout, stdout=subprocess.PIPE, stderr=None)
            
            # 2. Fecha a saída para não travar a memória
            process_rtl.stdout.close()
            
            # 3. O Python espera aqui até o SOX terminar os 30 segundos
            process_sox.communicate()
            
            # --- O EXORCISTA: Limpeza de Processos Fantasmas ---
            print("🧹 Limpando processos e liberando a antena...")
            process_rtl.terminate() # Manda o sinal para o rtl_fm morrer
            process_rtl.wait()      # Espera a confirmação de que ele realmente morreu
            time.sleep(1)           # Dá 1 segundo para o Windows respirar e resetar o USB
            # ---------------------------------------------------

            if os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 1000:
                print(f"✅ Gravação salva com sucesso: {self.output_file}")
                return True
            else:
                print("⚠️ Erro: Arquivo vazio ou corrompido.")
                return False
                
        except Exception as e:
            print(f"❌ Erro crítico: {e}")
            return False