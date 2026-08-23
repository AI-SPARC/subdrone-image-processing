# Odometria visual subaquática

Odometria visual monocular para vídeos gravados por um ROV. A partir de um
único vídeo, sem IMU, sensor de profundidade ou estéreo, o código estima o
caminho percorrido pelo veículo e desenha a trajetória.

O objetivo é comparar três estratégias clássicas de correspondência de pontos
(ORB, SIFT e fluxo óptico KLT) em condições subaquáticas, onde baixo contraste,
turbidez e dominância do azul degradam a detecção de features.

## Limitação de escala

Com uma única câmera e nenhum sensor auxiliar, a escala real do movimento é
matematicamente irrecuperável. O `cv2.recoverPose` devolve a translação como
vetor unitário, o que traz duas consequências:

- A trajetória é válida apenas a menos de escala. A forma pode estar correta,
  mas as distâncias não.
- Como todo passo é forçado a ter comprimento 1, a escala relativa entre passos
  também se perde. A forma só é fiel se a velocidade do veículo for
  aproximadamente constante entre keyframes.

Recuperar a escala exigiria informação adicional (estéreo, IMU, sensor de
pressão, ou um objeto de tamanho conhecido na cena) ou propagação de escala por
triangulação.

## Pipeline

1. `preprocess_videos.py` converte os vídeos de `dataset/raw/` para
   `dataset/processed/` via ffmpeg, padronizando a largura em 1280 px e a taxa
   em 30 fps, e removendo o áudio.
2. `utils.py` prepara cada frame: redimensiona para 960 px, converte para
   escala de cinza, aplica CLAHE e uma suavização leve. O CLAHE é o passo mais
   relevante aqui, dado o baixo contraste das cenas.
3. Os módulos em `methods/` extraem correspondências entre pares de frames.
4. `vo.py` calcula a matriz essencial com RANSAC, filtra os inliers e recupera
   rotação e translação com `recoverPose`, acumulando a pose global.
5. `main.py` percorre o vídeo em keyframes, acumula as posições e gera um HTML
   interativo com a vista 3D e três projeções 2D.
6. `plot_from_html.py` extrai as trajetórias do HTML e gera um PNG estático e
   um `.npz` reutilizável, permitindo replotar sem reprocessar o vídeo.

## Métodos comparados

| método | abordagem | comportamento observado |
|---|---|---|
| ORB | descritor binário, BFMatcher com distância de Hamming e ratio test de Lowe | diverge dos demais; pouco confiável em cena de baixa textura |
| SIFT | descritor de gradientes, BFMatcher L2 e ratio test de Lowe | mais estável; método preferido |
| KLT | cantos de Shi-Tomasi e fluxo óptico Lucas-Kanade com verificação ida e volta | bom com movimento suave |

## Resultados

A avaliação é qualitativa. Não há ground truth de pose para o vídeo utilizado,
o que impede calcular erro absoluto de trajetória (ATE) ou erro relativo (RPE).
O que se compara é a concordância entre métodos e a coerência da forma.

Na execução sobre o vídeo completo, com keyframe a cada 3 frames, os três
métodos produziram entre 876 e 939 poses. SIFT e KLT geram trajetórias
semelhantes na projeção de mapa (X–Z), enquanto o ORB se afasta
progressivamente dos outros dois. Como todos partem da origem e acumulam passos
unitários, a distância percorrida em unidades arbitrárias é aproximadamente
igual ao número de poses.

A deriva acumulada é visível: os métodos concordam no início do trajeto e se
separam ao longo do tempo, comportamento esperado de odometria quadro a quadro
sem otimização global ou fechamento de loop.

Obter uma medida quantitativa exigiria uma sequência com pose de referência,
seja de um dataset de benchmark, seja de uma filmagem com trajeto controlado.

## Matriz intrínseca

A matriz `K` é montada em `main.py` a partir de uma estimativa de distância
focal (`FX_SCALE`) e do centro da imagem. A matriz essencial e o `recoverPose`
são bastante sensíveis a `K`, e a refração entre água, vidro e ar altera o foco
efetivo. Para resultados confiáveis a câmera deve ser calibrada com tabuleiro
de xadrez, de preferência submersa, e `K` substituída pelos valores reais.

## Como rodar

Requer [uv](https://docs.astral.sh/uv/) e ffmpeg instalado. O caminho do
ffmpeg está definido em `preprocess_videos.py`.

```bash
uv sync
uv run python preprocess_videos.py
uv run python main.py
uv run python plot_from_html.py results/resultado_vo_6.html
```

Os resultados são gravados em `results/resultado_vo_N.html`, numerados
automaticamente.

## Parâmetros

Definidos no topo de `main.py`:

| parâmetro | descrição |
|---|---|
| `VIDEO_PATH` | vídeo processado a analisar |
| `FRAME_STEP` | distância em frames entre keyframes; valores maiores dão baseline maior e estimativa mais estável |
| `MAX_SKIP` | número de falhas consecutivas antes de ressincronizar o keyframe |
| `MAX_FRAMES` | limite de frames a processar; `None` processa o vídeo inteiro |
| `FX_SCALE` | fator da distância focal, com `fx = fy = FX_SCALE × largura` |
| `METHODS` | métodos a comparar |

A ressincronização do keyframe existe porque um trecho ruim do vídeo, como uma
virada brusca ou turbidez, pode fazer a estimativa falhar repetidamente. Sem
ela, o keyframe fica preso no passado e a odometria não se recupera pelo resto
do vídeo.

## Estrutura

| arquivo | papel |
|---|---|
| `main.py` | loop principal, seleção de keyframes, geração do HTML |
| `vo.py` | matriz essencial, `recoverPose` e acúmulo de pose |
| `utils.py` | preparo de cada frame |
| `preprocess_videos.py` | conversão dos vídeos via ffmpeg |
| `plot_from_html.py` | extração das trajetórias e plot estático |
| `methods/base.py` | interface comum dos métodos |
| `methods/orb.py` | correspondência por ORB |
| `methods/sift.py` | correspondência por SIFT |
| `methods/klt.py` | rastreamento por fluxo óptico |

## Possíveis evoluções

Propagação de escala por triangulação corrigiria a distorção de forma quando a
velocidade varia. Uma janela de otimização local ou a migração para ORB-SLAM3
trataria a deriva, já que o fechamento de loop corrige o acúmulo quando o
veículo revisita um ponto. Abordagens baseadas em aprendizado profundo, como
DROID-SLAM ou DPVO, tendem a ser mais robustas a baixa textura e turbidez, ao
custo de exigir GPU.
