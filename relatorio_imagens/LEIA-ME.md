# Imagens de resultados — Relatório Final PIBITI

Figuras reunidas para referência no relatório. Organizadas por frente de
trabalho. As legendas abaixo podem ser usadas como texto das figuras.

---

## odometria/

Resultados da odometria visual monocular sobre o vídeo `Black Eder_090419_
costelas do navio_2` (1280×720, 30 fps, 3.306 quadros). Keyframe a cada 3
quadros; 939 poses (ORB), 919 (SIFT) e 876 (KLT). **Todas as unidades são
arbitrárias**: a translação devolvida por `recoverPose` é unitária, portanto a
escala absoluta não é recuperável com uma única câmera.

| Arquivo | Descrição / legenda sugerida |
|---|---|
| `01_vista_mapa_XZ.png` | Vista de mapa (plano X–Z) das trajetórias estimadas por ORB, SIFT e KLT. Círculo marca o início (origem comum) e "X" o fim. SIFT e KLT percorrem o mesmo quadrante (+X, −Z); o ORB segue o quadrante oposto. |
| `02_projecoes_XZ_XY_ZY.png` | Três projeções ortogonais da trajetória (topo X–Z, frontal X–Y e lateral Z–Y), permitindo identificar o plano correspondente ao deslocamento horizontal. |
| `03_trajetoria_3d.png` | Trajetórias tridimensionais completas dos três métodos, escala arbitrária. |
| `04_divergencia_entre_metodos.png` | Distância euclidiana entre as trajetórias de cada par de métodos ao longo das poses. Quantifica a deriva acumulada: SIFT–KLT permanece baixa (média 54) enquanto os pares com ORB crescem continuamente (171 e 205). |
| `05_versao_inicial_1.png` a `05_versao_inicial_3.png` | Resultados da **versão inicial** do código, antes das correções: os três métodos partiam de pontos distintos do gráfico (a origem não era registrada) e as estimativas ainda não filtravam correspondências espúrias. Úteis como comparação "antes/depois". |

### Métricas comparativas (medidas em `gerar_figuras_relatorio.py`)

| Par de métodos | Distância média | Distância final |
|---|---|---|
| SIFT × KLT | 54 | 81 |
| ORB × SIFT | 171 | 256 |
| ORB × KLT | 205 | 326 |

Altura máxima alcançada no eixo Y: ORB 66, SIFT 161, KLT 194.

---

## segmentacao_corais/

Resultados do segmentador YOLO11n-seg treinado sobre o Coralscapes, com as 39
classes originais agrupadas em `coral_vivo`, `coral_branqueado` e `coral_morto`.
Correspondem ao **treino local em CPU** (fases 1 e 2); o retreino em GPU
(Colab, T4) fornecerá as métricas finais do relatório.

| Arquivo | Descrição / legenda sugerida |
|---|---|
| `01_matriz_confusao_normalizada.png` | Matriz de confusão normalizada no conjunto de validação. Evidencia a confusão entre coral morto e as demais classes. |
| `02_matriz_confusao.png` | Matriz de confusão em contagens absolutas. |
| `03_curva_precisao_recall_mascara.png` | Curva precisão–recall da máscara por classe. |
| `04_curva_f1_mascara.png` | Curva F1 da máscara em função do limiar de confiança. |
| `05_curvas_treino_fase1.png` | Curvas de perda e métricas da fase 1 (backbone congelado, `freeze=10`, AdamW). |
| `06_curvas_treino_fase2.png` | Curvas de perda e métricas da fase 2 (rede descongelada, taxa de aprendizado menor). O mAP50 da máscara ainda subia ao fim da fase, indicando subtreino. |
| `07_validacao_lote0_anotacao.jpg` / `08_validacao_lote0_predicao.jpg` | Comparação qualitativa: anotação de referência × predição do modelo (lote 0 da validação). |
| `09_validacao_lote1_anotacao.jpg` / `10_validacao_lote1_predicao.jpg` | Mesma comparação para o lote 1 da validação. |

---

## Como reproduzir as figuras da odometria

```bash
cd visual-odometry-sub
uv run python main.py                      # gera results/resultado_vo_N.html
uv run python plot_from_html.py results/resultado_vo_6.html   # gera o .npz
uv run python gerar_figuras_relatorio.py   # gera as figuras em alta resolução
```

## Observação

Não estão incluídas aqui as figuras da detecção de peixe-leão
(`segmentation/peixe-leao/runs/`), por corresponderem a trabalho desenvolvido
por outro integrante do laboratório. Estão disponíveis no repositório caso
sejam necessárias para contextualizar o pipeline unificado.
