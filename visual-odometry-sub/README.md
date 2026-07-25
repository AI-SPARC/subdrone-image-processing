# Visual Odometry Sub

Odometria visual **monocular** para vídeos gravados por um subdrone (ROV) embaixo
d'água. A partir de **um único vídeo** (sem IMU, sem sensores de profundidade, sem
estéreo), o código estima o caminho que o drone percorreu e desenha a trajetória.

## Limitação teórica importante (leia antes de interpretar resultados)

Com uma única câmera e nenhum sensor, **a escala real do movimento é
matematicamente irrecuperável**. O `cv2.recoverPose` sempre devolve a translação
`t` como um vetor **unitário** (norma 1). Consequências:

- A trajetória é válida apenas **a menos de escala** (a *forma* pode estar certa,
  os tamanhos/distâncias não).
- Como todo passo é forçado a ter comprimento 1, a **escala relativa** entre
  passos também se perde. A forma só é fiel se a velocidade do drone for
  aproximadamente constante entre keyframes.

Para escala correta seria preciso informação extra (estéreo, IMU/pressão,
altímetro, ou um objeto de tamanho conhecido na cena) ou propagação de escala por
triangulação (ver "Próximos passos").

## Como funciona (pipeline)

1. **Pré-processamento do vídeo** (`preprocess_videos.py`, via ffmpeg): converte os
   vídeos brutos em `dataset/raw/` para `dataset/processed/`, padronizando
   resolução (1280px de largura), FPS (30) e removendo o áudio.
2. **Pré-processamento por frame** (`utils.py::preprocess_frame`): redimensiona para
   960px, converte para escala de cinza, aplica **CLAHE** (equalização adaptativa,
   essencial em cena subaquática de baixo contraste) e uma leve suavização.
3. **Detecção/casamento de features** (`methods/`): para cada par de frames,
   extrai correspondências de pontos por um de três métodos.
4. **Estimativa de movimento** (`vo.py`): calcula a **matriz essencial** com RANSAC,
   filtra inliers e recupera rotação `R` e translação `t` (unitária) com
   `recoverPose`. Acumula a pose global (`R_total`, `t_total`).
5. **Trajetória + plot** (`main.py`): percorre o vídeo em **keyframes**, acumula as
   posições e gera um HTML interativo (Plotly) com 3D + 3 projeções 2D.
6. **Visualização estática** (`plot_from_html.py`): extrai as trajetórias do HTML e
   gera um PNG e um `.npz` reutilizável.

## Estrutura dos arquivos

| Arquivo | Papel |
|---|---|
| `preprocess_videos.py` | Converte vídeos de `dataset/raw` → `dataset/processed` (ffmpeg). |
| `utils.py` | `preprocess_frame`: resize + grayscale + CLAHE + blur. |
| `methods/base.py` | Interface `BaseMethod.get_matches(prev, cur) -> (pts1, pts2)`. |
| `methods/orb.py` | ORB + BFMatcher (Hamming) + **ratio test de Lowe**. |
| `methods/sift.py` | SIFT + BFMatcher (L2) + **ratio test de Lowe**. |
| `methods/klt.py` | `goodFeaturesToTrack` + fluxo óptico Lucas-Kanade + **forward-backward check**. |
| `vo.py` | `VisualOdometry`: essencial + `recoverPose` + acúmulo de pose, com guardas. |
| `main.py` | Loop principal, seleção de keyframes, geração do HTML. |
| `plot_from_html.py` | Extrai trajetórias do HTML → PNG + `.npz`. |

## Métodos de correspondência

- **ORB** — rápido, descritor binário. **Pouco confiável debaixo d'água** (baixo
  contraste/textura); tende a divergir. Mantido só para comparação.
- **SIFT** — mais robusto em cena subaquática; **método recomendado**.
- **KLT** — fluxo óptico; bom com movimento suave. Usa forward-backward check para
  descartar tracks ruins.

## Como rodar

Pré-requisitos: [uv](https://docs.astral.sh/uv/) e **ffmpeg** instalado
(ajuste o caminho do ffmpeg em `preprocess_videos.py` se necessário).

```bash
# 1. Instala dependências no ambiente virtual
uv sync

# 2. (uma vez) Converte os vídeos brutos
uv run python preprocess_videos.py

# 3. Roda a odometria visual e gera o HTML interativo em results/
uv run python main.py

# 4. (opcional) Gera PNG estático + .npz a partir de um HTML já criado
uv run python plot_from_html.py results/resultado_vo_6.html
```

Os resultados vão para `results/resultado_vo_N.html` (numerado
automaticamente), além de `results/trajetorias.png` e
`results/trajectories.npz` quando se usa o `plot_from_html.py`.

## Parâmetros (topo do `main.py`)

| Parâmetro | Descrição |
|---|---|
| `VIDEO_PATH` | Caminho do vídeo processado a analisar. |
| `FRAME_STEP` | Distância (em frames) entre keyframes. Maior = baseline maior/estimativa mais estável; menor = mais detalhe. |
| `MAX_SKIP` | Após N falhas seguidas, **ressincroniza** o keyframe (evita travar em trechos ruins). |
| `MAX_FRAMES` | Limita quantos frames processar (`None` = vídeo inteiro). Útil para testes rápidos. |
| `FX_SCALE` | `fx = fy = FX_SCALE * largura`. **Chute** da distância focal — o ideal é calibrar a câmera. |
| `METHODS` | Lista de métodos a comparar (`ORB`, `SIFT`, `KLT`). |

## Matriz intrínseca (K)

`K` é montada em `main.py` a partir de um chute de foco (`FX_SCALE`) e do centro da
imagem. A matriz essencial e o `recoverPose` são **muito sensíveis a `K`**, e a
refração água→vidro→ar altera o foco efetivo. Para resultados confiáveis, calibre a
câmera (de preferência dentro d'água, com tabuleiro de xadrez) e substitua `K`.

## Alterações recentes

Correções e melhorias sobre a primeira versão do pipeline:

- **`vo.py`**: filtra os **inliers** do RANSAC antes do `recoverPose`; descarta
  pares degenerados (matriz essencial não-3×3, poucos inliers, cheirality ruim);
  em caso de falha, **mantém a pose anterior** em vez de injetar ruído; passou a
  retornar uma flag de sucesso.
- **`methods/orb.py` e `methods/sift.py`**: trocado `crossCheck` por
  **knnMatch + ratio test de Lowe**, reduzindo muito os outliers.
- **`methods/klt.py`**: adicionado **forward-backward check** (rastreia ida e volta
  e descarta tracks inconsistentes).
- **`main.py`**:
  - A trajetória agora **começa sempre na origem** `(0,0,0)` — antes os métodos
    "nasciam" em pontos diferentes no gráfico.
  - Removido um **resize redundante** (o frame era reduzido e depois ampliado de
    volta, perdendo qualidade sem efeito prático).
  - Adicionada **seleção de keyframes** (`FRAME_STEP`) com **ressincronização**
    (`MAX_SKIP`) — corrige um bug em que um trecho ruim do vídeo (virada/turbidez)
    congelava o keyframe no passado e travava a odometria pelo resto do vídeo.
  - Processa o **vídeo inteiro** e gera **3D + 3 projeções 2D** (Topo X-Z, Frontal
    X-Y, Lateral Z-Y).
- **`plot_from_html.py`** (novo): extrai as trajetórias do HTML do Plotly e gera
  PNG estático + `.npz` reutilizável (replot sem reprocessar o vídeo).
- Dependência **matplotlib** adicionada.

## Observações dos resultados

- **SIFT** e **KLT** produzem trajetórias parecidas entre si no plano de mapa
  (X-Z) — bom indício de que capturam o movimento real.
- **ORB** diverge dos demais (confirma que não é adequado ao ambiente
  subaquático).
- Há **deriva (drift)** acumulada: os métodos concordam no início e se separam ao
  longo do tempo — limitação típica de VO quadro-a-quadro sem otimização global.

## Próximos passos (sem deep learning)

1. Fixar **SIFT** como método principal e remover/depriorizar o ORB.
2. **Propagação de escala por triangulação** (corrige distorção de forma quando a
   velocidade varia).
3. **Janela de otimização / bundle adjustment local**, ou migrar para
   **ORB-SLAM3** (tem fechamento de loop e corrige deriva se o drone revisitar um
   ponto).

Em último caso, abordagens de **deep learning** (ex.: DROID-SLAM, DPVO) tendem a
ser bem mais robustas a baixa textura/turbidez, ao custo de exigir GPU.
```
