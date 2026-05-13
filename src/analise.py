import os
import sys
import time
import re
import textwrap
import gc 
import tracemalloc
from collections import Counter
import pandas as pd
import ollama

# --- CONFIGURAÇÃO GRÁFICA SEGURA ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- BLINDAGEM DE CAMINHOS ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURRENT_DIR) == 'src':
    BASE_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
else:
    BASE_DIR = CURRENT_DIR

# Caminho Padrão (caso a interface não envie nenhum)
CAMINHO_CSV_PADRAO = os.path.join(BASE_DIR, 'dados', 'banco_transcricoes.csv')
CAMINHO_GRAFICOS = os.path.join(BASE_DIR, 'dados', 'graficos_tcc')

class CientistaSDR:
    def __init__(self, caminho_csv=None):
        print("🔬 A iniciar Analisador de Espectro (Motor Llama 3.2 Edge AI)...")
        os.makedirs(CAMINHO_GRAFICOS, exist_ok=True)
        
        # O MOTOR ULTRARRÁPIDO DA META
        self.modelo_llm = 'llama3.2:1b' 
        
        # Define qual o ficheiro CSV a ler (o escolhido pelo utilizador ou o padrão)
        self.caminho_csv = caminho_csv if caminho_csv else CAMINHO_CSV_PADRAO
        
        self.historico_tempos_a = []
        self.historico_tempos_b = []
        self.historico_memoria_a = [] 
        self.historico_memoria_b = [] 
        self.palavras_gerais = []
        self.categorias_detectadas = []
        self.entidades_gerais = []

    def limpar_texto(self, texto):
        return re.sub(r'[^\w\s]', '', str(texto).lower())

    def analise_quantitativa(self, texto):
        tracemalloc.start()
        inicio = time.time()
        
        texto_limpo = self.limpar_texto(texto)
        palavras = texto_limpo.split()
        
        stopwords = {
            'o', 'a', 'os', 'as', 'um', 'uma', 'uns', 'umas', 'de', 'do', 'da', 'dos', 'das',
            'em', 'no', 'na', 'nos', 'nas', 'para', 'com', 'que', 'e', 'é', 'se', 'por', 'pra', 'pro',
            'como', 'mais', 'mas', 'ou', 'eu', 'tu', 'ele', 'ela', 'nós', 'vós', 'eles', 'elas',
            'me', 'te', 'se', 'nos', 'vos', 'lhe', 'lhes', 'meu', 'minha', 'teu', 'tua', 'seu', 'sua',
            'nosso', 'nossa', 'este', 'esta', 'esse', 'essa', 'aquele', 'aquela', 'isso', 'isto', 'aquilo',
            'ser', 'estar', 'ter', 'fazer', 'ir', 'poder', 'ver', 'dar', 'saber', 'foi', 'era', 'são',
            'tem', 'têm', 'faz', 'vamos', 'vai', 'fui', 'sou', 'estou', 'está', 'estamos', 'estão',
            'ao', 'aos', 'pelo', 'pela', 'pelos', 'pelas', 'num', 'numa', 'nuns', 'numas',
            'qual', 'quais', 'quem', 'onde', 'quando', 'porquê', 'porque', 'aqui', 'ali', 'lá',
            'muito', 'pouco', 'tudo', 'nada', 'algo', 'alguém', 'ninguém', 'também', 'nem', 'já', 'até',
            'agora', 'sem', 'sobre', 'sob', 'entre', 'depois', 'antes', 'então', 'assim', 'só'
        }
        
        palavras_uteis = [p for p in palavras if p not in stopwords and len(p) > 2]
        self.palavras_gerais.extend(palavras_uteis)
        
        contagem = Counter(palavras_uteis)
        top_5 = [p[0] for p in contagem.most_common(5)] 
        
        tempo_gasto = time.time() - inicio
        _, pico_memoria = tracemalloc.get_traced_memory() 
        tracemalloc.stop()
        
        self.historico_tempos_a.append(tempo_gasto)
        self.historico_memoria_a.append(pico_memoria / 1024 / 1024) 
        return top_5, tempo_gasto

    def analise_qualitativa_llm(self, texto):
        tracemalloc.start() 
        inicio = time.time()
        
        prompt = f"""
        Analise a transcrição de rádio: "{texto}"
        
        Responda APENAS com esta estrutura exata, sem explicações extras:
        CATEGORIA: (Religioso, Jornalismo, Música, Trânsito, Comercial ou Outros)
        ENTIDADES: (Locais, Pessoas ou Marcas separadas por vírgula. Ou escreva "Nenhuma")
        RESUMO: (1 linha curta)
        """
        
        try:
            resposta = ollama.chat(
                model=self.modelo_llm, 
                messages=[{'role': 'user', 'content': prompt}],
                options={
                    'num_ctx': 1024,      
                    'num_predict': 150,   
                    'temperature': 0.1    
                }
            )
            resultado = resposta['message']['content'].strip()
            
            busca_cat = re.search(r'CATEGORIA:\s*(Religioso|Jornalismo|Música|Trânsito|Comercial|Outros)', resultado, re.IGNORECASE)
            cat = busca_cat.group(1).capitalize() if busca_cat else "Outros"
            self.categorias_detectadas.append(cat)
            
        except Exception as e:
            resultado = f"Erro IA: {e}"
            self.categorias_detectadas.append("Erro")
            
        tempo_gasto = time.time() - inicio
        _, pico_memoria = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        self.historico_tempos_b.append(tempo_gasto)
        self.historico_memoria_b.append(pico_memoria / 1024 / 1024) 
        return resultado

    def gerar_graficos_separados(self, df_completo):
        if not self.historico_tempos_a: return
            
        print("\n📈 A gerar as Imagens Otimizadas para o TCC...")
        sns.set_theme(style="whitegrid")
        prefixo_tempo = int(time.time())
        
        # =========================================================================
        # FIGURA 1: Trending Topics (Palavras)
        # =========================================================================
        if self.palavras_gerais:
            plt.figure(figsize=(8, 6))
            top_10 = Counter(self.palavras_gerais).most_common(10)
            sns.barplot(x=[p[1] for p in top_10], y=[p[0] for p in top_10], palette="magma", hue=[p[0] for p in top_10], legend=False)
            plt.title('Figura 1 - Estatística: Termos Mais Frequentes', fontsize=14, fontweight='bold')
            plt.xlabel('Frequência')
            plt.tight_layout()
            plt.savefig(os.path.join(CAMINHO_GRAFICOS, f"{prefixo_tempo}_fig1_palavras.png"), dpi=300)
            plt.close()

        # =========================================================================
        # FIGURA 2: Categorização (Pizza)
        # =========================================================================
        if self.categorias_detectadas:
            contagem_cat = Counter([c for c in self.categorias_detectadas if c != "Erro"])
            if contagem_cat:
                plt.figure(figsize=(8, 6))
                plt.pie(contagem_cat.values(), labels=contagem_cat.keys(), autopct='%1.1f%%', colors=sns.color_palette("Set2"))
                plt.title('Figura 2 - Contexto IA: Categorias de Rádio', fontsize=14, fontweight='bold')
                plt.tight_layout()
                plt.savefig(os.path.join(CAMINHO_GRAFICOS, f"{prefixo_tempo}_fig2_categorias.png"), dpi=300)
                plt.close()

        # =========================================================================
        # FIGURA 3: Entidades Detectadas (Agora com quebra de linha anti-corte)
        # =========================================================================
        if self.entidades_gerais:
            plt.figure(figsize=(9, 6)) 
            top_ents = Counter(self.entidades_gerais).most_common(7)
            nomes_formatados = [textwrap.fill(p[0], 15) for p in top_ents]
            valores = [p[1] for p in top_ents]
            
            sns.barplot(x=valores, y=nomes_formatados, palette="coolwarm", hue=nomes_formatados, legend=False)
            plt.title('Figura 3 - Contexto IA: Entidades e Alvos Detetados', fontsize=14, fontweight='bold')
            plt.xlabel('Menções')
            plt.tight_layout(pad=2.0) 
            plt.savefig(os.path.join(CAMINHO_GRAFICOS, f"{prefixo_tempo}_fig3_entidades.png"), dpi=300)
            plt.close()

        # =========================================================================
        # FIGURA 4: Desempenho Computacional (Tempo e RAM)
        # =========================================================================
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle('Figura 4 - Eficiência do Sistema Computacional', fontsize=14, fontweight='bold')
        eixo_x = ['CPU (Estatística)', 'Llama 3.2 (IA)']
        
        # Gráfico Tempo
        tempos = [sum(self.historico_tempos_a)/len(self.historico_tempos_a), sum(self.historico_tempos_b)/len(self.historico_tempos_b)]
        sns.barplot(x=eixo_x, y=tempos, ax=axes[0], palette="viridis", hue=eixo_x, legend=False)
        axes[0].set_title('Tempo Médio de Resposta', fontsize=12)
        axes[0].set_ylabel('Tempo (Segundos)')
        for i, v in enumerate(tempos): axes[0].text(i, v + (max(tempos)*0.05), f"{v:.3f} s", color='black', ha='center', fontweight='bold')

        # Gráfico Memória RAM
        memorias = [sum(self.historico_memoria_a)/len(self.historico_memoria_a), sum(self.historico_memoria_b)/len(self.historico_memoria_b)]
        sns.barplot(x=eixo_x, y=memorias, ax=axes[1], palette="crest", hue=eixo_x, legend=False)
        axes[1].set_title('Pico de Memória RAM Utilizada', fontsize=12)
        axes[1].set_ylabel('Memória (Megabytes)')
        for i, v in enumerate(memorias): axes[1].text(i, v + (max(memorias)*0.05), f"{v:.2f} MB", color='black', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(CAMINHO_GRAFICOS, f"{prefixo_tempo}_fig4_desempenho.png"), dpi=300)
        plt.close()

        # =========================================================================
        # FIGURA 5: Evolução Temporal 
        # =========================================================================
        plt.figure(figsize=(10, 5))
        df_completo['Data_Hora_DT'] = pd.to_datetime(df_completo['Data_Hora'], errors='coerce')
        df_valido = df_completo.dropna(subset=['Data_Hora_DT']).copy()
        
        if not df_valido.empty:
            df_valido['Hora_Formatada'] = df_valido['Data_Hora_DT'].dt.strftime('%H:00')
            contagem_horario = df_valido.groupby('Hora_Formatada').size().reset_index(name='Volume')
            
            sns.lineplot(data=contagem_horario, x='Hora_Formatada', y='Volume', marker='o', color='#d62728', linewidth=2.5, markersize=8)
            plt.title('Figura 5 - Evolução Temporal: Interceptações por Horário', fontsize=14, fontweight='bold')
            plt.xlabel('Horário do Dia')
            plt.ylabel('Volume de Transcrições (Eventos)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(CAMINHO_GRAFICOS, f"{prefixo_tempo}_fig5_linha_do_tempo.png"), dpi=300)
        plt.close()

        del df_completo
        del df_valido
        gc.collect()

        print(f"✅ Figuras e Métricas guardadas com sucesso!")
        try: os.startfile(CAMINHO_GRAFICOS)
        except: pass

    def executar_analise(self, limite=5):
        # Utiliza o ficheiro escolhido pelo utilizador
        if not os.path.exists(self.caminho_csv):
            raise Exception(f"Banco de dados não encontrado em {self.caminho_csv}")
            
        df_completo = pd.read_csv(self.caminho_csv) 
        ultimos = df_completo.tail(limite) 
        
        nome_ficheiro = os.path.basename(self.caminho_csv)
        print(f"\n📡 MONITORAMENTO INTELIGENTE (Analisando ficheiro: {nome_ficheiro})\n" + "="*80)
        
        tabela_alertas = []
        
        for _, linha in ultimos.iterrows():
            print(f"📻 ALVO: {linha['Frequencia_MHz']} MHz")
            
            top_palavras, t_a = self.analise_quantitativa(linha['Texto_Transcrito'])
            analise_ia = self.analise_qualitativa_llm(linha['Texto_Transcrito'])
            
            entidades_extraidas = "Nenhuma"
            resumo_extraido = "Sem resumo"
            categoria_extraida = self.categorias_detectadas[-1] if self.categorias_detectadas else "Desconhecida"
            
            linhas_ia = analise_ia.split('\n')
            for l in linhas_ia:
                texto_linha = l.strip()
                if texto_linha:
                    if re.match(r'(?i)^ENTIDADES:', texto_linha):
                        entidades_extraidas = texto_linha.split(':', 1)[1].strip()
                        if entidades_extraidas.lower() not in ['nenhuma', 'nenhum', 'erro', 'n/a']:
                            ents = [e.strip() for e in entidades_extraidas.split(',')]
                            self.entidades_gerais.extend(ents)
                            
                    elif re.match(r'(?i)^RESUMO:', texto_linha):
                        resumo_extraido = texto_linha.split(':', 1)[1].strip()
            
            tabela_alertas.append({
                "Data/Hora": linha['Data_Hora'],
                "Frequência": f"{linha['Frequencia_MHz']} MHz",
                "Categoria": categoria_extraida,
                "Entidades": entidades_extraidas,
                "Resumo": resumo_extraido
            })
            
            print(f"   ✓ Analisado com sucesso.")
            
        self.gerar_graficos_separados(df_completo)
        
        if tabela_alertas:
            df_alertas = pd.DataFrame(tabela_alertas)
            caminho_tabela = os.path.join(BASE_DIR, 'dados', 'relatorio_alertas_tcc.csv')
            df_alertas.to_csv(caminho_tabela, index=False, encoding='utf-8-sig')
            
        gc.collect()

if __name__ == "__main__":
    CientistaSDR().executar_analise(limite=5)