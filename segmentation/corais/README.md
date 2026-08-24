# Segmentação de corais e análise de saúde

Segmentação semântica de corais em imagens subaquáticas e estimativa do estado
de saúde de cada colônia. Faz parte do estudo de análise de imagem para o ROV
de monitoramento da APA Costa dos Corais.

O problema é dividido em duas etapas com responsabilidades distintas:

| pergunta | quem responde |
|---|---|
| onde há coral na imagem? | modelo YOLO-seg |
| o coral está vivo ou morto? | modelo YOLO-seg (usa textura) |
| o coral vivo está saudável ou branqueando? | análise de cor calibrada |

A separação não é arbitrária. Medições descritas adiante mostram que a cor
média do pixel não distingue coral morto de coral vivo, porque esqueleto
coberto de alga reflete luz de forma parecida com tecido pigmentado. Já a
distinção entre saudável e branqueado é bem capturada pela cor, desde que o
sinal escolhido seja o brilho e não a saturação.

## Dataset

Os dados vêm do **Coralscapes** (Sauder et al.), o primeiro conjunto de
segmentação semântica densa de recifes de coral: 2.075 imagens, 174 mil
polígonos e 39 classes anotadas por especialistas, sob licença CC-BY-4.0.
As imagens são do Mar Vermelho, a 5–7 m de profundidade.

O que o torna adequado aqui é que ele distingue o estado do coral (`alive`,
`bleached`, `dead`) por gênero, o que fornece tanto as classes de treino
quanto uma verdade de referência para validar a análise de cor.

As 39 classes originais são agrupadas em 3 pelo mapa `map_coral.json`:

| classe | nome | IDs do Coralscapes |
|---|---|---|
| 0 | `coral_vivo` | 6, 17, 21, 22, 25, 27, 28, 31, 34, 36 |
| 1 | `coral_branqueado` | 4, 16, 19, 33 |
| 2 | `coral_morto` | 3, 20, 23, 32, 37 |

Água, areia, peixes, mergulhadores e equipamento não entram no mapa e viram
fundo. Após a conversão para o formato YOLO-seg:

| split | imagens | polígonos | vivo | branqueado | morto |
|---|---|---|---|---|---|
| train | 1.517 | 15.274 | 12.353 | 551 | 2.370 |
| valid | 166 | 1.728 | 1.094 | 129 | 505 |
| test | 392 | 3.132 | 2.363 | 142 | 627 |

Nem toda imagem sobrevive ao filtro de área: no treino, 1.392 das 1.517 ficam
com pelo menos um polígono, e as demais entram como fundo.

A classe `coral_branqueado` representa 3,6% das instâncias de treino.
Esse desbalanceamento é relevante na avaliação: o mAP global pode ficar alto
enquanto a classe rara vai mal, então o recall por classe deve ser observado
separadamente.

As imagens são exportadas com 1024 px de largura. O treino usa 640 px, e
manter os 2048 px originais aumentava o custo de memória e o tempo de
validação sem benefício, já que as máscaras previstas são reescaladas para a
resolução original a cada época.

O filtro `--min-area 1500` é aplicado sobre a máscara já exportada, em
1024×512, e descarta manchas menores que cerca de 0,29% da área da imagem —
aproximadamente 39×39 px, ou 24×24 px na resolução de treino.

O valor é absoluto, em pixels, então a fração de área que ele corta depende da
resolução de exportação. Trocar `--max-width` sem regerar os rótulos deixa o
conjunto inconsistente: as coordenadas YOLO são normalizadas e continuam
carregando sem erro, mas passam a descrever um recorte de instâncias diferente
do que o filtro produziria naquela resolução.

## Análise de saúde por cor

O módulo `coral_health.py` converte a região da máscara para HSV e classifica
o estado em saudável, pálido ou branqueado. A hipótese inicial era a
intuitiva: coral saudável tem pigmento e portanto alta saturação, enquanto
coral branqueado é claro e sem cor.

Os rótulos do Coralscapes permitiram testar essa hipótese. O script
`gerar_calibracao.py` recorta cada instância usando a máscara e zera o fundo,
de modo que a cor medida venha apenas do coral. Foram usados 300 recortes de
coral vivo, 241 de branqueado e 300 de morto.

A separabilidade entre vivo e branqueado, medida pelo d de Cohen:

| sinal | vivo | branqueado | d de Cohen |
|---|---|---|---|
| saturação (S) | 0,333 | 0,305 | 0,23 |
| brilho (V) | 0,529 | 0,793 | 2,46 |
| índice V·(1−S) | 0,352 | 0,557 | 1,74 |

A saturação praticamente não distingue as duas classes. A explicação é física:
o véu de luz retroespalhada pela água dessatura toda a cena por igual, o que
retira da saturação o poder de discriminação. O esqueleto de carbonato
exposto, porém, continua refletindo mais luz que o tecido pigmentado, e o
contraste de brilho sobrevive à atenuação.

O efeito na classificação é grande:

| regra de decisão | acurácia vivo × branqueado |
|---|---|
| saturação, limiares heurísticos | 31,2% |
| saturação, limiares otimizados | 58,4% |
| índice V·(1−S), otimizado | 82,1% |
| brilho, otimizado | 89,3% |

Com os limiares heurísticos originais, apenas 9% dos corais branqueados reais
eram detectados. Por isso `coral_health.py` expõe uma chave `regra` que
seleciona o sinal de decisão, e `calibrate_health.py` compara as três famílias
em vez de assumir uma.

### A faixa intermediária

O Coralscapes não possui classe "pálido" ou "estressado". Ao otimizar apenas
para duas classes, a busca em grade colapsa essa faixa, encostando o limiar de
pálido no de branqueado.

A faixa foi reaberta deliberadamente em `bri_pale = 0,62`. O custo foi medido:
a detecção de branqueamento não muda (201 acertos em 241 nas duas
configurações), 51 corais saudáveis passam a ser marcados como pálidos e a
acurácia nominal cai de 89,3% para 80,2%. Os pontos perdidos são inteiramente
corais saudáveis reclassificados como "atenção", e não branqueamento não
detectado. Para monitoramento de recife, esse viés conservador é preferível à
acurácia nominal maior.

### Correção de cor

Corrigir a cor antes de analisar parecia vantajoso, mas o teste não confirmou:
a acurácia foi de 89,3% na imagem crua contra 86,9% com correção automática.
A correção redistribui os canais para recuperar saturação, enquanto o sinal
que estava funcionando era o brilho, e ela ainda introduz variação extra entre
imagens.

O módulo `underwater_color_correction.py` continua útil para visualização e
para material fora d'água. A conclusão específica é que os limiares devem ser
calibrados no mesmo pipeline usado em produção: calibrar em imagem crua e
aplicar em imagem corrigida invalida os limiares.

### Coral morto

O diagnóstico de `gerar_calibracao.py` sobre os recortes rotulados:

```
saudavel     n=300  ->  saudavel=78%   palido=17%  branqueado=5%
branqueado   n=241  ->  branqueado=83% palido=9%   saudavel=8%
morto        n=300  ->  saudavel=72%   palido=17%  branqueado=11%
```

72% do coral morto é classificado como saudável pela cor. A informação que
separa as duas condições está na textura e na estrutura, não na cor média do
pixel, e nenhum ajuste de limiar resolve isso. Daí a escolha de treinar o
modelo com 3 classes e deixar a cor atuar apenas dentro do que o modelo já
identificou como coral vivo.

Na inferência isso é aplicado por `aplicar_classe_do_modelo()`: quando o
modelo classifica a instância como `coral_morto`, o veredito da cor é
registrado em `categoria_por_cor` para auditoria mas não decide, e
`summarize_reef` exclui coral morto das médias de saúde, reportando
`area_morta_pct` separadamente.

## Limitações

- A calibração é local. Os limiares valem para a câmera, o sítio e o pipeline
  de cor usados. Outra profundidade ou outra água exigem recalibrar.
- A faixa "pálido" não tem verdade de referência, sendo uma escolha
  conservadora documentada e não um valor medido.
- A anotação do Coralscapes é semântica, não de instância. Colônias vizinhas
  do mesmo gênero podem formar um único polígono, o que limita a contagem de
  colônias mas não a estimativa de cobertura por área.
- Os valores RGB de `CORALWATCH_REFERENCE` são aproximados. Para rigor, o
  cartão CoralWatch deve ser fotografado com a própria câmera.
- O valor absoluto do índice de saúde é frágil. O acompanhamento do mesmo
  coral ao longo do tempo, com câmera e pipeline fixos, é mais confiável que
  o julgamento de uma imagem isolada.
- `remove_veil` fica desligado por padrão: em cenas onde o coral é o objeto
  mais escuro, o estimador de retroespalhamento subtrai sinal real e
  supersatura o resultado.

## Ambiente

O ambiente virtual precisa ficar em um caminho curto. Em caminhos longos a
instalação do PyTorch falha no Windows com `WinError 206`, porque alguns
arquivos de licença ultrapassam o limite de 260 caracteres.

```powershell
python -m venv C:\Users\guilh\venvs\coral
C:\Users\guilh\venvs\coral\Scripts\Activate.ps1
pip install -r requirements.txt
```

O treino roda em CPU. A GPU Intel Arc integrada desta máquina conclui treinos
em subconjuntos pequenos mas falha no dataset completo, com erros do backend
Level Zero (`UR_RESULT_ERROR_OUT_OF_RESOURCES` ou violação de acesso). Foram
testados sem sucesso dois drivers, lotes de 2 a 8, AMP ligado e desligado, o
caminho `foreach` dos otimizadores, 0 a 4 workers e duas resoluções de
imagem. Para tentar mesmo assim, use `--device xpu`.

Em CPU, uma época completa leva cerca de 8 minutos.

## Uso

Preparar o dataset:

```powershell
python exportar_coralscapes.py
python convert_annotations.py mask2yolo --masks train\masks --out train\labels --class-map map_coral.json --min-area 1500
python convert_annotations.py validate --labels train\labels --nc 3
```

Calibrar os limiares de saúde:

```powershell
python gerar_calibracao.py --split valid --max-por-classe 300
python calibrate_health.py fit --dir calib
```

Treinar e avaliar:

```powershell
python train_phase1.py
python train_phase2.py
python evaluate.py val --model trained_models\yolo_coral_seg\best.pt --data data.yaml --split test
```

Inferência, isolada ou junto com a detecção de peixe-leão:

```powershell
python predict_coral.py --model trained_models\yolo_coral_seg\best.pt --source imagens\ --out saida\ --erode-px 3
python ..\pipeline_unificado.py --lionfish-model ..\peixe-leao\trained_models\yolov12_lionfish\best.pt --coral-model trained_models\yolo_coral_seg\best.pt --source img.jpg --out saida_unificada\
```

O parâmetro `--erode-px` reduz a máscara alguns pixels antes de medir a cor.
Sem isso, uma máscara que vaze para a areia clara puxa a média para o branco e
gera falso branqueamento.

## Arquivos

| arquivo | função |
|---|---|
| `coral_health.py` | análise de saúde por cor (HSV, NumPy) |
| `underwater_color_correction.py` | correção de cor subaquática |
| `convert_annotations.py` | conversão COCO/máscara/bbox para YOLO-seg, validação |
| `exportar_coralscapes.py` | download e exportação do Coralscapes |
| `gerar_calibracao.py` | monta o conjunto de calibração a partir das máscaras |
| `calibrate_health.py` | ajuste dos limiares por busca em grade |
| `train_phase1.py` | treino com backbone congelado |
| `train_phase2.py` | ajuste fino e exportação ONNX/TorchScript |
| `train_common.py` | seleção de dispositivo e localização de checkpoints |
| `predict_coral.py` | inferência com overlay e relatório JSON |
| `evaluate.py` | métricas, curvas de treino e agregação de saúde |
| `data.yaml` | configuração do dataset |
| `map_coral.json` | mapa das 39 classes do Coralscapes para 3 |

Os módulos de análise têm autoteste que roda sem dataset, GPU ou PyTorch:

```powershell
python coral_health.py
python underwater_color_correction.py
python convert_annotations.py selftest
python calibrate_health.py selftest
python evaluate.py selftest
```

## Referências

- Sauder, J. et al. *The Coralscapes Dataset: Semantic Scene Understanding in
  Coral Reefs.* <https://huggingface.co/datasets/EPFL-ECEO/coralscapes>
- Ancuti, C. O. et al. *Color Balance and Fusion for Underwater Image
  Enhancement.* IEEE Transactions on Image Processing, 2018.
- CoralWatch, *Coral Health Chart.* <https://coralwatch.org>
- Jocher, G. et al. *Ultralytics YOLO.* <https://github.com/ultralytics/ultralytics>
