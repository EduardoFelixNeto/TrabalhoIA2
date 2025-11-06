# -------------------------------------------------------------------------------------------------- # Linha separadora para organização visual
# Detector de círculos usando AG (Algoritmo Genético) + RBFNN (Rede de Funções de Base Radial).     # Breve descrição do objetivo do script
# Melhorias principais:                                                                              # Lista de melhorias implementadas
#  - Limiar adaptativo (Otsu) sobre a saída contínua da rede RBF, com fallback fixo se necessário.   # Explica a estratégia de binarização da saída
#  - Função de aptidão combina IoU + α*Revocação − β*Falsos Positivos − γ*Esparsidade − δ*Diversidade.# Explica como a qualidade de um indivíduo é medida
#  - Penalidade de diversidade para evitar centros RBF colapsados (sobrepostos no mesmo círculo).     # Evita que várias unidades detectem a mesma região
#  - Esparsidade mais relaxada para não perder círculos pequenos/mais tênues.                         # Ajuste fino para sensibilidade a alvos pequenos
# -------------------------------------------------------------------------------------------------- # Linha separadora para organização visual

import os                                                                                             # Fornece funções para manipular o sistema de arquivos (pastas, caminhos, criação)
import math                                                                                           # Funções matemáticas (exp, log, sqrt etc.) usadas em gaussianas e Kåsa
import random                                                                                         # Gerador de números aleatórios da biblioteca padrão (para reprodutibilidade/AG)
from dataclasses import dataclass                                                                     # Facilita criar classes de dados imutáveis/imutáveis com pouco código
from typing import Tuple, List                                                                        # Tipos auxiliares para anotações (legibilidade e IDEs)

import numpy as np                                                                                    # Biblioteca NumPy para operações de alto desempenho com matrizes/arrays
from PIL import Image, ImageOps                                                                       # Biblioteca Pillow para carregar imagens e aplicar transformações simples

# ---------------- Parâmetros de Configuração ----------------                                       # Bloco que concentra todos os parâmetros ajustáveis do experimento
DIRETORIO_IMAGENS = "./images_2"                                                                      # Pasta de onde será lida a primeira imagem para detectar círculos
DIRETORIO_SAIDA   = "./output"                                                                        # Pasta onde os resultados (imagens geradas) serão salvos
REDUZIR_PARA      = 256                                                                               # Limite do maior lado da imagem para acelerar o processamento (downscale)
QTD_UNIDADES_RBF  = 14                                                                                # Número de unidades RBF (centros gaussianos) no cromossomo (pode ter sobras)
TAMANHO_POPULACAO = 80                                                                                # Quantos indivíduos compõem cada geração do Algoritmo Genético
GERACOES          = 120                                                                                # Quantas gerações o AG irá evoluir (mais gerações = mais tempo, melhor ajuste)
TORNEIO_K         = 3                                                                                 # Tamanho do torneio na seleção (competem K indivíduos e vence o melhor)
PROB_CRUZAMENTO   = 0.8                                                                               # Probabilidade de realizar cruzamento entre dois pais (blend crossover)
TAXA_MUTACAO      = 0.16                                                                              # Probabilidade de cada gene sofrer mutação (ruído gaussiano)
SIGMA_MUT_POS     = 7.0                                                                               # Intensidade (desvio padrão) do ruído na mutação das posições (X e Y) em pixels
SIGMA_MUT_SIGMA   = 0.08                                                                              # Intensidade (desvio padrão) do ruído na mutação do log(sigma) (escala gaussiana)
SIGMA_MUT_AMP     = 0.12                                                                              # Intensidade (desvio padrão) do ruído na mutação das amplitudes das unidades RBF
SEMENTE_ALEATORIA = 123                                                                               # Semente fixa para tornar os resultados reproduzíveis entre execuções

# Pesos/coeficientes dos termos da função de aptidão                                                  # Explicita a contribuição de cada termo na qualidade do indivíduo
PESO_FP           = 0.28                                                                              # Penaliza falsos positivos (pixels preditos como círculo onde não há círculo)
PESO_ESPAR        = 0.004                                                                             # Penaliza amplitudes totais altas (estimula uso parcimonioso das unidades)
PESO_REVOC        = 0.28                                                                              # Recompensa a revocação (cobrir o máximo dos pixels reais de círculo)
PESO_DIVERS       = 0.06                                                                              # Penaliza unidades com centros muito próximos (estimula diversidade espacial)
FRAC_ESCALA_RHO   = 0.12                                                                              # Define o “alcance” da interação entre centros como fração do menor lado da imagem
FALLBACK_LIMIAR   = 0.36                                                                              # Limiar fixo usado se o Otsu retornar valor não confiável (degen.)
INTERVALO_VIESES  = (-0.1, 1.0)                                                                       # Intervalo permitido para o bias (deslocamento) da saída da rede

random.seed(SEMENTE_ALEATORIA)                                                                        # Inicializa o gerador aleatório padrão com a semente definida
np.random.seed(SEMENTE_ALEATORIA)                                                                     # Inicializa o gerador aleatório do NumPy com a mesma semente

@dataclass                                                                                            # Indica que esta classe é uma “dataclass” (gera __init__, __repr__, etc.)
class GenomaRBF:
    """                                                                                                # Início da docstring da classe (descrição em português)
    Estrutura que representa um indivíduo do Algoritmo Genético para a rede RBF.                       # Explica o propósito do contêiner
    - 'genes' guarda todos os parâmetros: para cada unidade RBF temos [cx, cy, log_sigma, amplitude],  # Detalha a organização do vetor de genes
      repetido QTD_UNIDADES_RBF vezes, seguido de 1 gene de bias global no final.                      # Esclarece a presença do bias (termo constante)
    - 'largura' e 'altura' guardam as dimensões da imagem para clamping e construção de grades.         # Mostra a utilidade das dimensões da imagem
    """
    genes: np.ndarray                                                                                  # Array 1D com os genes (parâmetros) do indivíduo
    largura: int                                                                                       # Largura (em pixels) da imagem usada como alvo
    altura: int                                                                                        # Altura (em pixels) da imagem usada como alvo

def garantir_diretorios() -> None:
    """Cria a pasta de saída se ela ainda não existir, evitando erros ao salvar arquivos."""           # Docstring simples da função utilitária
    os.makedirs(DIRETORIO_SAIDA, exist_ok=True)                                                        # Cria DIRETORIO_SAIDA de forma idempotente (não estoura se já existir)

def carregar_primeira_imagem(caminho: str) -> Image.Image:
    """Lê a primeira imagem válida (por extensão) encontrada no diretório informado e retorna em L (cinza)."""  # Explica o comportamento da leitura
    extensoes_validas = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")                               # Lista de extensões aceitas para filtrar arquivos de imagem
    for nome_arquivo in os.listdir(caminho):                                                             # Percorre todos os nomes de arquivos da pasta
        if nome_arquivo.lower().endswith(extensoes_validas):                                             # Verifica se o arquivo possui uma extensão de imagem válida
            return Image.open(os.path.join(caminho, nome_arquivo)).convert("L")                          # Abre o arquivo e converte para tons de cinza (modo 'L')
    raise FileNotFoundError(f"Nenhuma imagem encontrada em {caminho}.")                                  # Se não encontrar nenhuma imagem, aborta com erro claro

def reduzir_escala(imagem: Image.Image, lado_maximo: int) -> Image.Image:
    """Reduz o tamanho da imagem mantendo a proporção se o maior lado exceder 'lado_maximo'."""         # Docstring explicando o downscale
    largura, altura = imagem.size                                                                        # Obtém largura e altura originais para calcular o fator de redução
    escala = max(largura, altura) / lado_maximo if max(largura, altura) > lado_maximo else 1.0           # Calcula a razão de redução (>=1); 1 significa “não reduzir”
    return (imagem.resize((int(round(largura/escala)),                                                   # Redimensiona a imagem para (largura/escala, altura/escala), arredondados
                          int(round(altura/escala))),
            Image.LANCZOS)[0] if escala > 1 else imagem)                                                  # Usa filtro LANCZOS para qualidade; se não precisa reduzir, retorna a original

def binarizar_mascara_alvo(imagem: Image.Image) -> np.ndarray:
    """Gera uma máscara binária onde áreas originalmente escuras (círculos) viram 1 e fundo claro vira 0."""  # Explica a finalidade da binarização
    imagem_invertida = ImageOps.invert(imagem)                                                             # Inverte as cores para facilitar o limiar (preto→alto, branco→baixo)
    matriz_float = np.asarray(imagem_invertida, dtype=np.float32) / 255.0                                  # Converte para array float normalizado em [0,1]
    limiar_base = max(0.15, min(0.85, matriz_float.mean() * 0.8))                                          # Define um limiar inicial com base na média, limitado para evitar extremos
    return (matriz_float > limiar_base).astype(np.float32)                                                 # Cria e retorna a máscara binária (1 se acima do limiar, senão 0)

def inicializar_genoma(largura: int, altura: int, qtd_unidades: int) -> GenomaRBF:
    """Cria um indivíduo aleatório: posiciona centros, define escalas (sigmas) e amplitudes dentro de faixas razoáveis."""  # Explica a inicialização
    lista_genes = []                                                                                       # Vetor acumulador temporário para montar o cromossomo completo
    raio_min, raio_max = 0.03*min(largura, altura), 0.35*min(largura, altura)                              # Define faixa de raios plausíveis como frações do menor lado
    for _ in range(qtd_unidades):                                                                          # Repete o processo para cada unidade RBF
        cx = np.random.uniform(0, largura)                                                                 # Sorteia a coordenada X do centro dentro dos limites da imagem
        cy = np.random.uniform(0, altura)                                                                  # Sorteia a coordenada Y do centro dentro dos limites da imagem
        log_sigma = np.random.uniform(math.log(raio_min), math.log(raio_max))                              # Sorteia log(sigma) para garantir sigma positivo ao exponenciar
        amplitude = np.clip(np.random.normal(0.8, 0.25), 0.0, 1.5)                                         # Sorteia amplitude com distribuição normal, limitada a [0, 1.5]
        lista_genes.extend([cx, cy, log_sigma, amplitude])                                                 # Acrescenta os quatro genes desta unidade ao cromossomo
    bias = np.random.uniform(*INTERVALO_VIESES)                                                            # Sorteia o viés global (bias) dentro do intervalo permitido
    lista_genes.append(bias)                                                                               # Adiciona o gene de bias ao fim do vetor de genes
    return GenomaRBF(np.array(lista_genes, dtype=np.float32), largura, altura)                             # Converte para np.array float32 e retorna o dataclass do genoma

def decodificar_parametros(genoma: GenomaRBF):
    """Lê o vetor de genes e retorna (lista_de_unidades, bias), já com sigma = exp(log_sigma)."""         # Explica o retorno e a conversão de escala
    g = genoma.genes                                                                                      # Atalho local para o array de genes (melhora legibilidade)
    parametros_unidades = []                                                                              # Lista que receberá tuplas (cx, cy, sigma, amplitude)
    for i in range(0, 4*QTD_UNIDADES_RBF, 4):                                                             # Percorre o cromossomo em blocos de 4 posições por unidade
        cx, cy, log_sigma, amplitude = g[i:i+4]                                                           # Extrai os quatro valores da unidade corrente
        parametros_unidades.append((cx, cy, math.exp(log_sigma), amplitude))                              # Converte log_sigma em sigma e guarda a tupla
    bias = g[4*QTD_UNIDADES_RBF]                                                                          # Captura o último gene como bias global da rede
    return parametros_unidades, bias                                                                       # Retorna os parâmetros das unidades e o bias global

def propagar_rbfn(genoma: GenomaRBF) -> np.ndarray:
    """Calcula a saída contínua da rede RBF em cada pixel (aplica soma ponderada de gaussianas + sigmoide)."""  # Explica o cálculo da saída
    parametros_unidades, bias = decodificar_parametros(genoma)                                             # Obtém lista de unidades (centros/sigmas/amplitudes) e o bias
    h, w = genoma.altura, genoma.largura                                                                   # Lê dimensões usadas para montar a grade (y,x)
    grade_y, grade_x = np.mgrid[0:h, 0:w]                                                                  # Cria duas matrizes: uma com índices de linha (y) e outra de coluna (x)
    grade_x = grade_x.astype(np.float32)                                                                   # Converte grade de x para float32 (desempenho/consistência numérica)
    grade_y = grade_y.astype(np.float32)                                                                   # Converte grade de y para float32 (desempenho/consistência numérica)
    saida = np.full((h, w), bias, dtype=np.float32)                                                        # Inicia o mapa de saída preenchido com o bias (ponto de partida)
    for cx, cy, sigma, amplitude in parametros_unidades:                                                   # Itera por cada unidade RBF
        dx = grade_x - cx                                                                                  # Calcula deslocamento horizontal de cada pixel até o centro da unidade
        dy = grade_y - cy                                                                                  # Calcula deslocamento vertical de cada pixel até o centro da unidade
        gauss = np.exp(-(dx*dx + dy*dy) / (2.0 * (sigma*sigma) + 1e-9))                                    # Avalia a gaussiana radial (evita divisão por zero com 1e-9)
        saida += amplitude * gauss                                                                         # Acrescenta a contribuição ponderada dessa unidade ao mapa contínuo
    return 1.0 / (1.0 + np.exp(-saida))                                                                    # Aplica sigmoide para restringir a saída ao intervalo [0,1]

def intersecao_sobre_uniao(mascara_pred: np.ndarray, mascara_alvo: np.ndarray) -> float:
    """Compute IoU (Intersection over Union) entre a máscara predita e a máscara alvo."""                  # Docstring sobre a métrica IoU
    inter = np.logical_and(mascara_pred, mascara_alvo).sum()                                               # Conta quantos pixels são 1 nas duas máscaras simultaneamente (interseção)
    uniao = np.logical_or(mascara_pred, mascara_alvo).sum()                                                # Conta quantos pixels são 1 em pelo menos uma das máscaras (união)
    return (inter / uniao) if uniao else (1.0 if inter == 0 else 0.0)                                      # Retorna IoU; em caso de união zero, trata o caso limite corretamente

def revocacao(mascara_pred: np.ndarray, mascara_alvo: np.ndarray) -> float:
    """Mede a fração de pixels positivos do alvo que foram corretamente recuperados pela predição (Recall).""" # Docstring sobre a métrica de recall
    verdadeiros_positivos = np.logical_and(mascara_pred, mascara_alvo).sum()                               # Conta quantos pixels positivos do alvo foram marcados como positivos
    positivos_alvo = mascara_alvo.sum()                                                                     # Conta quantos pixels positivos existem na máscara alvo
    return (verdadeiros_positivos / positivos_alvo) if positivos_alvo else 1.0                             # Se não há positivos no alvo, retorna 1.0 por convenção

def limiar_otsu(valores: np.ndarray) -> float:
    """Calcula automaticamente o melhor limiar em [0,1] maximizando a separação entre classes (método de Otsu).""" # Explica o algoritmo de Otsu
    hist, bordas = np.histogram(valores.ravel(), bins=256, range=(0,1))                                     # Monta histograma de 256 níveis sobre a imagem/saída contínua
    hist = hist.astype(np.float64)                                                                          # Converte contagens para float64 para operações suaves
    prob = hist / (hist.sum() + 1e-12)                                                                      # Normaliza para obter distribuição de probabilidade (soma=1)
    omega = np.cumsum(prob)                                                                                 # Calcula a probabilidade acumulada até cada bin (classe 0)
    mu = np.cumsum(prob * np.arange(256))                                                                   # Calcula a média acumulada ponderada pelos níveis de cinza
    mu_total = mu[-1]                                                                                       # Média total dos níveis (último elemento da cumulativa)
    sigma_entre = (mu_total*omega - mu)**2 / (omega*(1.0 - omega) + 1e-12)                                  # Variância entre classes (critério de Otsu) por limiar
    indice = np.nanargmax(sigma_entre)                                                                      # Escolhe o limiar que maximiza a separação entre classes
    limiar = (indice + 0.5) / 256.0                                                                         # Converte o índice do bin para um valor contínuo em [0,1]
    return float(limiar)                                                                                    # Retorna como float nativo do Python (não NumPy)

def penalidade_diversidade(genoma: GenomaRBF) -> float:
    """Calcula quão próximos estão os centros das RBFs; maior proximidade ⇒ maior penalidade (evita colapso).""" # Docstring sobre a penalidade
    parametros_unidades, _ = decodificar_parametros(genoma)                                                # Obtém os parâmetros das unidades RBF do indivíduo
    centros = np.array([[cx, cy] for (cx, cy, _, _) in parametros_unidades], dtype=np.float32)             # Extrai os pares (cx, cy) para todos os centros em um array
    if centros.shape[0] < 2:                                                                               # Se existe menos de 2 centros, não há pares para comparar
        return 0.0                                                                                         # Sem penalização quando não há pares
    rho = FRAC_ESCALA_RHO * min(genoma.largura, genoma.altura)                                             # Define o alcance de “interação” com base no menor lado da imagem
    inv_dois_rho2 = 1.0 / (2.0 * rho * rho + 1e-9)                                                         # Pré-calcula 1/(2*rho^2) com termo de estabilidade numérica
    penalidade = 0.0                                                                                       # Inicializa o acumulador da penalidade total
    for i in range(centros.shape[0]):                                                                       # Itera sobre o primeiro índice do par de centros
        for j in range(i+1, centros.shape[0]):                                                              # Itera sobre o segundo índice (j>i) para evitar pares duplicados
            dist2 = ((centros[i] - centros[j])**2).sum()                                                    # Calcula distância euclidiana ao quadrado entre os dois centros
            penalidade += math.exp(-dist2 * inv_dois_rho2)                                                  # Soma contribuição gaussiana (quanto mais perto, maior a penalidade)
    total_pares = centros.shape[0] * (centros.shape[0]-1) / 2.0                                             # Número total de pares únicos combinatórios
    return penalidade / (total_pares + 1e-9)                                                                # Normaliza pela quantidade de pares para manter escala estável

def aptidao(genoma: GenomaRBF, mascara_alvo: np.ndarray) -> float:
    """Combina métricas (IoU/Recall) e penalidades (FP/Esparsidade/Diversidade) para medir a qualidade do indivíduo.""" # Docstring sobre a função objetivo
    mapa_saida = propagar_rbfn(genoma)                                                                     # Gera a saída contínua [0,1] da rede RBF para todos os pixels
    limiar = limiar_otsu(mapa_saida)                                                                       # Tenta separar automaticamente “círculo” vs “fundo” via Otsu
    if not (0.05 <= limiar <= 0.95):                                                                       # Se o Otsu sugerir um limiar muito extremo (suspeito)
        limiar = FALLBACK_LIMIAR                                                                           # Usa um valor fixo de reserva para não travar a evolução
    mascara_predita = (mapa_saida >= limiar)                                                               # Converte a saída contínua em máscara binária com o limiar escolhido

    valor_iou = intersecao_sobre_uniao(mascara_predita, mascara_alvo)                                      # Mede a sobreposição relativa entre predição e alvo (qualidade geométrica)
    valor_revoc = revocacao(mascara_predita, mascara_alvo)                                                 # Mede a fração de pixels do alvo corretamente recuperados (sensibilidade)
    falsos_positivos = np.logical_and(mascara_predita, ~mascara_alvo.astype(bool)).sum()                   # Conta quantos pixels foram marcados como círculo sem serem alvo
    taxa_fp = falsos_positivos / (mascara_alvo.size + 1e-9)                                                # Normaliza FP pelo total de pixels para ter escala comparável
    amplitudes = genoma.genes[3:4*QTD_UNIDADES_RBF:4]                                                       # Extrai todas as amplitudes (um a cada 4 genes, começando no índice 3)
    esparsidade = np.abs(amplitudes).sum() / (QTD_UNIDADES_RBF + 1e-9)                                      # Usa a soma das amplitudes como proxy de “quantas unidades efetivas”
    diversidade = penalidade_diversidade(genoma)                                                            # Mede proximidade média entre centros (queremos menor)

    return (valor_iou                                                                                       # Termo principal: quão bem a máscara predita coincide em área com o alvo
            + PESO_REVOC*valor_revoc                                                                        # Bônus por cobrir mais pixels do alvo (recall)
            - PESO_FP*taxa_fp                                                                               # Penalidade por marcar pixels onde não há círculo (precisão)
            - PESO_ESPAR*esparsidade                                                                        # Penalidade por usar “muita” amplitude total (parsimonioso)
            - PESO_DIVERS*diversidade)                                                                      # Penalidade por centros muito próximos (evita colapso de modo)

def limitar_genes(genes: np.ndarray, largura: int, altura: int) -> None:
    """Garante que cada parâmetro permaneça em faixas válidas (clamping) para evitar valores absurdos."""  # Docstring sobre o clamping
    for i in range(0, 4*QTD_UNIDADES_RBF, 4):                                                              # Percorre cada unidade (blocos de 4 genes)
        genes[i]   = np.clip(genes[i],   0, largura-1)                                                     # X do centro deve estar dentro da imagem [0, largura-1]
        genes[i+1] = np.clip(genes[i+1], 0, altura-1)                                                      # Y do centro deve estar dentro da imagem [0, altura-1]
        genes[i+2] = np.clip(genes[i+2],                                                                   # log_sigma também é limitado para evitar sigmas ridiculamente pequenos/grandes
                              math.log(0.02*min(largura, altura)),                                         # Limite inferior do raio (2% do menor lado)
                              math.log(0.5*min(largura, altura)))                                          # Limite superior do raio (50% do menor lado)
        genes[i+3] = np.clip(genes[i+3], 0.0, 2.0)                                                         # Amplitude limitada [0, 2] (evita saturação ou sinais invertidos)
    genes[4*QTD_UNIDADES_RBF] = np.clip(genes[4*QTD_UNIDADES_RBF], *INTERVALO_VIESES)                      # Bias global limitado ao intervalo pré-definido

def selecao_torneio(populacao: List[GenomaRBF], lista_aptidoes: List[float]) -> GenomaRBF:
    """Escolhe um indivíduo por torneio: amostra K índices e retorna a melhor aptidão dentre eles (exploração/exploração).""" # Docstring sobre o método de seleção
    indices = np.random.choice(len(populacao), size=TORNEIO_K, replace=False)                             # Seleciona aleatoriamente K indivíduos distintos para competir
    melhor_indice = max(indices, key=lambda j: lista_aptidoes[j])                                         # Entre os K, escolhe aquele com maior valor de aptidão
    return GenomaRBF(populacao[melhor_indice].genes.copy(),                                               # Retorna uma cópia do cromossomo do vencedor (evita mutação in-place)
                     populacao[melhor_indice].largura,                                                    # Mantém a mesma largura da imagem do indivíduo original
                     populacao[melhor_indice].altura)                                                     # Mantém a mesma altura da imagem do indivíduo original

def cruzamento(pai1: GenomaRBF, pai2: GenomaRBF):
    """Realiza um cruzamento do tipo “blend” (combinação convexa gene a gene) entre dois pais e retorna dois filhos.""" # Docstring do operador de cruzamento
    g1, g2 = pai1.genes.copy(), pai2.genes.copy()                                                          # Copia os vetores de genes para não alterar os pais originais
    if np.random.rand() < PROB_CRUZAMENTO:                                                                 # Com probabilidade PROB_CRUZAMENTO, efetua o cruzamento
        alpha = np.random.rand(g1.size)                                                                     # Gera um vetor alpha de [0,1] do mesmo tamanho dos genes (peso por gene)
        filho_a = alpha * g1 + (1-alpha) * g2                                                               # Primeiro filho: mistura ponderada g1/g2
        filho_b = alpha * g2 + (1-alpha) * g1                                                               # Segundo filho: mistura complementar g2/g1
        g1, g2 = filho_a, filho_b                                                                           # Substitui os cromossomos pelos filhos gerados
    return (GenomaRBF(g1, pai1.largura, pai1.altura),                                                       # Retorna objeto GenomaRBF do primeiro filho (mantém dimensões)
            GenomaRBF(g2, pai2.largura, pai2.altura))                                                       # Retorna objeto GenomaRBF do segundo filho (mantém dimensões)

def mutacao(individuo: GenomaRBF) -> None:
    """Aplica pequenas perturbações gaussiana nos genes com certa probabilidade para manter variabilidade na população.""" # Docstring do operador de mutação
    g = individuo.genes                                                                                    # Atalho para o vetor de genes a ser modificado
    largura, altura = individuo.largura, individuo.altura                                                  # Dimensões usadas no clamping após a mutação
    for i in range(0, 4*QTD_UNIDADES_RBF, 4):                                                              # Percorre cada unidade RBF (4 genes por unidade)
        if np.random.rand() < TAXA_MUTACAO: g[i]   += np.random.normal(0, SIGMA_MUT_POS)                  # Se sorteado, move levemente o centro X
        if np.random.rand() < TAXA_MUTACAO: g[i+1] += np.random.normal(0, SIGMA_MUT_POS)                  # Se sorteado, move levemente o centro Y
        if np.random.rand() < TAXA_MUTACAO: g[i+2] += np.random.normal(0, SIGMA_MUT_SIGMA)                # Se sorteado, altera discretamente o log_sigma
        if np.random.rand() < TAXA_MUTACAO: g[i+3] += np.random.normal(0, SIGMA_MUT_AMP)                  # Se sorteado, altera discretamente a amplitude
    if np.random.rand() < TAXA_MUTACAO: g[4*QTD_UNIDADES_RBF] += np.random.normal(0, 0.05)                # Também pode ajustar um pouco o bias global
    limitar_genes(g, largura, altura)                                                                      # Após mutar, garante que os genes continuam em faixas seguras

def componentes_conectados(mascara: np.ndarray):
    """Varre a máscara binária e separa regiões 1’s contíguas (4-vizinhos) em componentes distintos (lista de pixels).""" # Docstring do rotulador simples
    h, w = mascara.shape                                                                                   # Obtém altura e largura para limites de varredura
    visitado = np.zeros_like(mascara, dtype=bool)                                                          # Matriz booleana que marca se o pixel já foi explorado
    componentes = []                                                                                       # Lista final de componentes, cada um contendo os pontos (y,x)
    for y in range(h):                                                                                     # Percorre cada linha da imagem/máscara
        for x in range(w):                                                                                 # Percorre cada coluna da imagem/máscara
            if mascara[y, x] and not visitado[y, x]:                                                       # Entra quando encontra um pixel 1 ainda não explorado
                fila = [(y, x)]                                                                            # Inicializa uma pilha (usando lista) com o pixel semente
                visitado[y, x] = True                                                                      # Marca o pixel semente como visitado para não reprocessar
                pontos = []                                                                                # Lista temporária para coletar todos os pixels do componente
                while fila:                                                                                # Continua enquanto houver vizinhos a explorar
                    cy, cx = fila.pop()                                                                    # Retira o último ponto adicionado (busca em profundidade simples)
                    pontos.append((cy, cx))                                                                # Adiciona o ponto atual à lista do componente
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:                                            # Considera vizinhança 4-conexa (cima, baixo, esquerda, direita)
                        ny, nx = cy + dy, cx + dx                                                          # Calcula as coordenadas do vizinho em questão
                        if 0 <= ny < h and 0 <= nx < w and mascara[ny, nx] and not visitado[ny, nx]:       # Verifica limites, se é 1 e se ainda não foi visitado
                            visitado[ny, nx] = True                                                        # Marca o vizinho como visitado para não entrar em loop
                            fila.append((ny, nx))                                                          # Agenda o vizinho para também expandir sua vizinhança
                componentes.append(np.array(pontos, dtype=np.int32))                                       # Converte a lista de pontos do componente em array e armazena
    return componentes                                                                                     # Retorna a lista de todos os componentes encontrados

def ajustar_circulo_kasa(pontos: np.ndarray):
    """Ajusta um círculo pelo método de Kåsa usando mínimos quadrados; retorna (cx, cy, r) ou NaN se insuficiente.""" # Docstring explicando Kåsa
    if pontos.shape[0] < 3:                                                                                # Verifica se há pontos suficientes (um círculo precisa de >=3)
        return float('nan'), float('nan'), float('nan')                                                    # Sem pontos suficientes ⇒ devolve NaN para indicar falha
    x = pontos[:,1].astype(np.float64)                                                                     # Extrai as coordenadas de coluna (x) convertendo para float64
    y = pontos[:,0].astype(np.float64)                                                                     # Extrai as coordenadas de linha (y) convertendo para float64
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])                                                       # Monta a matriz A do sistema linear da formulação de Kåsa
    b = x*x + y*y                                                                                          # Monta o vetor b correspondente a x² + y²
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)                                                            # Resolve A·sol ≈ b via mínimos quadrados (estável numericamente)
    cx, cy, c = sol                                                                                        # Desempacota os parâmetros estimados (centro e termo constante)
    r = math.sqrt(max(c + cx*cx + cy*cy, 0.0))                                                             # Converte a solução no raio r, garantindo argumento ≥0 antes da raiz
    return cx, cy, r                                                                                       # Retorna centro (cx, cy) e raio r do círculo ajustado

def executar_uma_vez():
    """Pipeline completo: lê imagem, evolui AG, binariza saída, extrai componentes e ajusta círculos; salva visualizações.""" # Explica o fluxo fim-a-fim
    garantir_diretorios()                                                                                  # Cria a pasta de saída, se ainda não existir, para evitar falhas ao salvar
    imagem = carregar_primeira_imagem(DIRETORIO_IMAGENS)                                                   # Lê a primeira imagem válida encontrada no diretório configurado
    imagem = reduzir_escala(imagem, REDUZIR_PARA)                                                          # Reduz a imagem para acelerar o cálculo mantendo a proporção
    largura, altura = imagem.size                                                                          # Captura a largura e altura finais (após possível redução)
    mascara_alvo = binarizar_mascara_alvo(imagem)                                                          # Constrói uma máscara binária: objeto escuro (círculo) = 1, fundo = 0

    populacao = [inicializar_genoma(largura, altura, QTD_UNIDADES_RBF)                                     # Cria uma lista de indivíduos aleatórios (genes diferentes)
                 for _ in range(TAMANHO_POPULACAO)]                                                        # Repete até atingir o tamanho da população configurada
    aptidoes = [aptidao(ind, mascara_alvo) for ind in populacao]                                           # Avalia a função de aptidão de cada indivíduo inicial
    melhor_indice = int(np.argmax(aptidoes))                                                               # Localiza o índice do indivíduo com maior aptidão (melhor solução atual)
    melhor_individuo, melhor_aptidao = populacao[melhor_indice], aptidoes[melhor_indice]                   # Guarda referência ao melhor indivíduo e seu valor de aptidão
    print(f"Aptidão inicial (best): {melhor_aptidao:.4f}")                                                 # Exibe no console a aptidão do melhor indivíduo da geração 0

    for ger in range(1, GERACOES+1):                                                                       # Inicia o laço principal do AG, percorrendo cada geração
        nova_populacao = []                                                                                # Começa uma nova lista que representará a próxima geração
        elite = GenomaRBF(populacao[melhor_indice].genes.copy(), largura, altura)                          # Implementa elitismo: clona o melhor indivíduo para preservar progresso
        nova_populacao.append(elite)                                                                       # Garante que a solução atual (elite) não se perca no próximo passo
        while len(nova_populacao) < TAMANHO_POPULACAO:                                                     # Preenche o restante da população com filhos gerados
            p1 = selecao_torneio(populacao, aptidoes)                                                      # Seleciona o primeiro pai via torneio (exploração equilibrada)
            p2 = selecao_torneio(populacao, aptidoes)                                                      # Seleciona o segundo pai via torneio (independente do primeiro)
            f1, f2 = cruzamento(p1, p2)                                                                    # Realiza cruzamento (com probabilidade PROB_CRUZAMENTO) gerando dois filhos
            mutacao(f1); mutacao(f2)                                                                       # Aplica mutação gene a gene para manter diversidade (evitar estagnação)
            nova_populacao.extend([f1, f2])                                                                # Adiciona os dois filhos à lista da próxima geração
        populacao = nova_populacao[:TAMANHO_POPULACAO]                                                     # Se houve excesso por paridade, trunca a lista ao tamanho correto
        aptidoes = [aptidao(ind, mascara_alvo) for ind in populacao]                                       # Recalcula a aptidão de todos os indivíduos da nova geração
        melhor_indice = int(np.argmax(aptidoes))                                                           # Atualiza o índice do melhor indivíduo atual
        if aptidoes[melhor_indice] > melhor_aptidao:                                                       # Se a nova geração trouxe alguém melhor que o melhor histórico
            melhor_aptidao = aptidoes[melhor_indice]                                                       # Atualiza o valor da melhor aptidão já observada
            melhor_individuo = populacao[melhor_indice]                                                    # Atualiza a referência para o novo melhor indivíduo
        if ger % 10 == 0:                                                                                  # A cada 10 gerações, faz um log para acompanhar a convergência
            print(f"Geração {ger:03d} | melhor={melhor_aptidao:.4f}")                                      # Mostra a geração atual e a melhor aptidão acumulada até aqui

    mapa_final = propagar_rbfn(melhor_individuo)                                                           # Propaga o melhor indivíduo ao final da evolução (saída contínua final)
    limiar_final = limiar_otsu(mapa_final)                                                                 # Tenta separar fundo/objeto na saída final usando Otsu
    if not (0.05 <= limiar_final <= 0.95):                                                                 # Se o limiar final é suspeito (muito extremo)
        limiar_final = FALLBACK_LIMIAR                                                                     # Recorre ao limiar fixo de segurança para binarização
    mascara_predita = (mapa_final >= limiar_final)                                                         # Cria a máscara binária final (pixels com prob >= limiar viram 1)

    componentes = componentes_conectados(mascara_predita)                                                  # Segmenta a máscara em regiões contíguas para ajustar círculos por região
    area_minima = 0.0004 * (largura * altura)                                                               # Define uma área mínima (0,1% da imagem) para filtrar ruído muito pequeno
    componentes = [c for c in componentes if c.shape[0] >= area_minima]                                    # Mantém apenas componentes com área suficiente para serem círculos reais

    circulos = []                                                                                          # Lista onde acumularemos os círculos válidos encontrados
    for comp in componentes:                                                                               # Itera por cada componente conectado remanescente
        cx, cy, r = ajustar_circulo_kasa(comp)                                                             # Ajusta um círculo via método de Kåsa usando os pontos do componente
        if not np.isnan(cx) and r > 2:                                                                     # Valida o resultado (centro definido) e ignora raios muito pequenos
            circulos.append((cx, cy, r))                                                                   # Armazena o círculo detectado (centro e raio)

    print(f"Detectados {len(circulos)} círculo(s).")                                                       # Loga a quantidade final de círculos detectados
    for i, (cx, cy, r) in enumerate(circulos, 1):                                                          # Percorre a lista para dar feedback detalhado por círculo
        print(f"  {i}: centro=({cx:.1f}, {cy:.1f}), r={r:.1f}")                                            # Imprime centro (x,y) e raio com 1 casa decimal

    import matplotlib.pyplot as plt                                                                        # Importa Matplotlib aqui (adiado) para só pesar quando necessário
    theta = np.linspace(0, 2*np.pi, 256)                                                                   # Cria 256 ângulos uniformes para desenhar a circunferência suavemente

    os.makedirs(DIRETORIO_SAIDA, exist_ok=True)                                                            # Garante mais uma vez que a pasta de saída existe (robustez)
    figura1 = plt.figure()                                                                                 # Abre uma nova figura para a sobreposição de detecções
    plt.imshow(imagem, cmap="gray")                                                                        # Mostra a imagem (já reduzida) em tons de cinza como plano de fundo
    for cx, cy, r in circulos:                                                                             # Para cada círculo detectado
        xs = cx + r*np.cos(theta)                                                                          # Calcula as coordenadas X dos pontos da circunferência
        ys = cy + r*np.sin(theta)                                                                          # Calcula as coordenadas Y dos pontos da circunferência
        plt.plot(xs, ys, linewidth=2)                                                                      # Desenha a linha do círculo com espessura visível
    plt.title(f"Detecções: {len(circulos)}")                                                               # Coloca um título indicando quantos círculos foram encontrados
    caminho_deteccoes = os.path.join(DIRETORIO_SAIDA, "deteccoes.png")                                     # Define o caminho do arquivo de saída para a imagem de detecções
    figura1.savefig(caminho_deteccoes, bbox_inches="tight")                                                # Salva a figura com a sobreposição dos círculos detectados
    plt.close(figura1)                                                                                     # Fecha a figura para liberar memória/recursos

    figura2 = plt.figure(figsize=(9,3))                                                                    # Abre outra figura para um comparativo em 3 painéis (alvo/saída/máscara)
    plt.subplot(1,3,1); plt.imshow(mascara_alvo, cmap="gray");                                            # Painel 1: mostra a máscara alvo (o que queremos detectar)
    plt.title("Máscara alvo"); plt.axis("off")                                                             # Define o título do painel e esconde os eixos
    plt.subplot(1,3,2); plt.imshow(mapa_final, cmap="gray");                                               # Painel 2: mostra a saída contínua da rede RBF (probabilidades)
    plt.title(f"Saída RBFNN (limiar={limiar_final:.2f})"); plt.axis("off")                                 # Informa o limiar usado e esconde os eixos para foco visual
    plt.subplot(1,3,3); plt.imshow(mascara_predita, cmap="gray");                                          # Painel 3: mostra a máscara binária final após limiarização
    plt.title("Máscara predita"); plt.axis("off")                                                          # Define título e remove eixos para apresentação limpa
    caminho_viz = os.path.join(DIRETORIO_SAIDA, "resultado_evolucao.png")                                  # Caminho do arquivo de saída para a figura comparativa
    figura2.savefig(caminho_viz, bbox_inches="tight")                                                      # Salva a figura comparativa com margens justas
    plt.close(figura2)                                                                                     # Fecha a figura para liberar memória/recursos

    print(f"Arquivos salvos em: {caminho_deteccoes} | {caminho_viz}")                                      # Mensagem final indicando onde os resultados foram gravados

if __name__ == "__main__":                                                                                 # Garante que o bloco abaixo só rode quando o script for executado diretamente
    executar_uma_vez()                                                                                     # Dispara todo o pipeline uma única vez (leitura→AG→binarização→ajuste→salva)
