# Relatório Final PIBITI — rascunho para o SIGAA

Texto para copiar nos campos do Relatório Final de Iniciação Científica.
Cada bloco traz a contagem de caracteres (incluindo espaços e quebras de linha)
no momento em que o arquivo foi gerado. O SIGAA corta no limite; não ultrapasse.

Campos já preenchidos no sistema (não editar):
Discente: 21110819 — Guilherme de Oliveira Costa
Orientador: Richard James Ladle
Título do plano: Estudo de aplicações de análise de imagem para o contexto de um ROV subaquático
Projeto: PVCB4256-2025 — Desenvolvimento de submarino ROV para o monitoramento de longo prazo da APA Costa dos Corais (PELD_CCAL) e produção de vídeos educativos infantis – FASE 3

Os trechos marcados com `[A PREENCHER]` dependem das métricas do retreino no Colab.

---

## Tipo de pesquisa

Pesquisa Tecnológica

(O edital é PIBITI. A entrega é um pipeline de software, não um resultado teórico isolado.)

## Progresso da pesquisa

Concluída

(Refere-se à bolsa. O projeto PELD_CCAL continua; a iniciação deste plano se encerra com este relatório.)

## ODS da Agenda 2030 (marcar)

- Conservação e uso sustentável dos oceanos, dos mares e dos recursos marinhos (ODS 14)
- Construir infraestruturas resilientes, promover a industrialização inclusiva e sustentável e fomentar a inovação (ODS 9)
- Tomar medidas urgentes para combater a mudança climática e seus impactos (ODS 13) — opcional, pelo recorte de branqueamento

---

## Resumo (português, 1500, usado 1060)

Este trabalho desenvolveu duas aplicações de análise de imagem para um ROV de monitoramento da APA Costa dos Corais: segmentação de corais com estimativa de saúde e odometria visual monocular. O plano original foi realinhado após a detecção do peixe-leão ter sido concluída por outro membro do laboratório. A segmentação usa YOLO11n-seg sobre o Coralscapes, com 39 classes agrupadas em vivo, branqueado e morto. A saúde do coral vivo é lida pela cor da máscara. Calibração com recortes rotulados mostrou que o brilho separa vivo de branqueado (d de Cohen = 2,46; acurácia 89,3%), enquanto a saturação não o faz (d = 0,23), por causa do véu de luz da água. Coral morto não se distingue pela cor (72% lido como saudável) e permanece a cargo do modelo. A odometria compara ORB, SIFT e fluxo óptico KLT em vídeo real, com CLAHE. Sem pose de referência, a avaliação é qualitativa: SIFT e KLT geram trajetórias semelhantes; o ORB diverge. Com uma câmera, a escala absoluta é irrecuperável. [A PREENCHER: uma frase com mAP50 e recall por classe do modelo retreinado.]

## Palavras-chave (70, usado 44)

ROV, corais, odometria visual, branqueamento

## Title (inglês, 200, usado 86)

Image analysis for an underwater ROV: coral segmentation and monocular visual odometry

## Abstract (inglês, 1500, usado 1086)

This work developed two image-analysis applications for an ROV used in long-term monitoring of the Costa dos Corais APA: coral segmentation with health estimation, and monocular visual odometry. The original plan was realigned after lionfish detection had already been completed by another lab member. Segmentation uses YOLO11n-seg on Coralscapes, with 39 classes grouped into live, bleached and dead coral. Health of live coral is read from mask colour. Calibration on labelled crops showed that brightness separates live from bleached tissue (Cohen's d = 2.46; 89.3% accuracy), whereas saturation does not (d = 0.23), because of the water backscatter veil. Dead coral is not distinguishable by mean colour (72% read as healthy) and is left to the model. Odometry compares ORB, SIFT and KLT optical flow on real underwater video, with CLAHE. Without ground-truth pose, evaluation is qualitative: SIFT and KLT produce similar trajectories; ORB diverges. With a single camera, absolute scale is unrecoverable. [FILL IN: one sentence with mAP50 and per-class recall after GPU retraining.]

## Keywords (70, usado 44)

ROV, coral reefs, visual odometry, bleaching

---

## Introdução (4000, usado 3297)

A Área de Proteção Ambiental Costa dos Corais é o maior recife costeiro raso do Brasil e está sujeita a branqueamento, espécies invasoras e pressão antrópica. O projeto PELD_CCAL desenvolve um ROV para monitoramento de longo prazo nesse recife. Uma vez que o veículo passa a filmar de forma sistemática, o valor científico do material depende de extração automática: localizar o que há no fundo e estimar por onde o veículo se moveu.

Este plano de trabalho trata dessas duas extrações. A primeira é a segmentação de corais e a leitura do estado de saúde de cada colônia. A segunda é a odometria visual monocular, isto é, a estimativa da trajetória a partir de uma única câmera, sem GPS, IMU nem sensor de profundidade. O plano original (Edital 03 PIBITI UFAL 2025-2026) previa familiarização com processamento de imagem, estudo dos problemas ópticos do meio submerso, implementação com OpenCV e aprendizado profundo, validação e relatórios. No recorte parcial o escopo foi ajustado: outro aluno já havia implementado a detecção do peixe-leão, e a contribuição deste trabalho passou a ser o incremento com segmentação de corais e a odometria para navegação.

O branqueamento ocorre quando o coral, sob estresse térmico, expulsa as zooxantelas e expõe o esqueleto de carbonato, que reflete mais luz (CoralWatch). Monitorar esse processo em imagem exige, primeiro, saber onde está o coral e, depois, interpretar a cor do tecido. Conjuntos públicos com anotação densa e estado de saúde por colônia são recentes. O Coralscapes (Sauder et al., 2025) reúne 2.075 imagens do Mar Vermelho, 174 mil polígonos e 39 classes, incluindo vivo, branqueado e morto por gênero, o que permite tanto treinar um segmentador quanto calibrar a leitura de cor contra verdade de referência.

A navegação submersa não dispõe de GNSS. Soluções acústicas existem, mas são caras e pouco precisas a curta distância, faixa em que a câmera é informativa (Ferrera et al., 2019). A odometria visual estima o movimento entre quadros por correspondência de pontos. Em água, porém, a turbidez, a absorção seletiva, a iluminação artificial e a baixa textura degradam os descritores clássicos. Azhmukhamedov et al. (2021) argumentam a necessidade de VO local em ROVs e testam ORB-SLAM3 em vídeo de monitoramento. Zhang, Ila e Kneip (2018) tratam VO estéreo robusta a iluminação irregular. Ferrera et al. (2019) mostram que o fluxo óptico resiste melhor à turbidez do que o casamento de descritores, e o conjunto AQUALOC (Ferrera et al., 2019) oferece sequências com pose de referência. Nordfeldt-Fiol (2022) observa ganho ao trocar o detector do LIBVISO2 por SIFT/SURF em fundos colonizados por algas. Desses trabalhos saem duas decisões deste estudo: comparar ORB, SIFT e KLT no mesmo vídeo, e não tratar o ORB como escolha óbvia só porque é rápido.

As duas frentes compartilham o tipo de entrada (vídeo subaquático) e o destino (o ROV), mas não o mesmo modelo. A segmentação opera em imagem anotada e devolve máscaras; a odometria opera em vídeo sem pose verdadeira e devolve uma curva a menos de escala. Mantê-las separadas permite evoluir uma sem invalidar a outra. O objetivo deste relatório é descrever os métodos, confrontar os resultados com a literatura e registrar as limitações que o próprio experimento tornou mensuráveis.

## Metodologia (4000, usado 3167)

O trabalho divide-se em dois pipelines independentes, ambos em Python, executados sobre material pré-gravado.

Na segmentação, as imagens e máscaras vêm do Coralscapes (licença CC-BY-4.0), baixadas via Hugging Face e exportadas com 1024 px de largura. As 39 classes originais foram agrupadas em três pelo mapa `map_coral.json`: coral_vivo, coral_branqueado e coral_morto. Água, areia, peixes e equipamento viram fundo. Máscaras indexadas foram convertidas em polígonos YOLO-seg, descartando instâncias menores que cerca de 0,07% da área. O conjunto de treino ficou com 1.517 imagens e 26.911 polígonos, dos quais apenas 4% são branqueados. O modelo é o YOLO11n-seg pré-treinado no COCO. A fase 1 congela o backbone (freeze=10) e ajusta a cabeça com AdamW; a fase 2 descongela a rede com taxa menor. Imagens de treino a 640 px; aumentos de cor deliberadamente moderados para não corromper a leitura de saúde. A GPU Intel integrada desta máquina conclui subconjuntos pequenos, mas falha no conjunto completo (erros do backend Level Zero); o treino local rodou em CPU. Um retreino em GPU (Colab, T4) com mais épocas está em curso para as métricas finais. A avaliação usa o split de teste (392 imagens, 6.357 instâncias), não visto no treino, e reporta precisão, recall e mAP50 da máscara por classe.

A saúde do coral vivo é medida depois da máscara. Converte-se a região para HSV em NumPy e classifica-se em saudável, pálido ou branqueado. A hipótese inicial (saturação alta = pigmento) foi testada em recortes cujo fundo foi zerado pela máscara: 300 vivos, 241 branqueados e 300 mortos do split de validação. A separabilidade (d de Cohen) foi 0,23 para a saturação, 2,46 para o brilho e 1,74 para o índice V·(1−S). Uma busca em grade escolheu a regra de brilho. A classe “pálido” não existe no Coralscapes; o limiar intermediário foi reaberto de propósito, aceitando queda de acurácia nominal em troca de um viés conservador (saudável marcado como atenção, e não branqueamento perdido). Correção de cor antes da classificação piorou o resultado (89,3% na imagem crua contra 86,9% com correção), porque redistribui canais quando o sinal útil era o brilho. Coral morto não entra nessa regra: 72% dele é lido como saudável pela cor, de modo que o veredito de morte fica com o segmentador.

Na odometria, vídeos de `dataset/raw/` são padronizados com ffmpeg (1280 px, 30 fps, sem áudio). Cada quadro é redimensionado para 960 px, convertido para cinza, equalizado com CLAHE (clipLimit=3, grade 8×8) e suavizado. Três correspondências alimentam o mesmo núcleo: ORB (3000 pontos, Hamming, ratio test 0,75), SIFT (L2, ratio 0,70) e KLT (cantos Shi-Tomasi e Lucas-Kanade, verificação ida e volta com limiar de 1 px). A matriz essencial é estimada por RANSAC; `recoverPose` devolve translação unitária, acumulada na pose global. Keyframe a cada 3 quadros; após duas falhas seguidas o keyframe é ressincronizado. A matriz intrínseca usa `fx = fy = largura` e o centro da imagem: é um chute, não uma calibração. O vídeo de teste é uma sequência real junto a costelas de naufrágio (Black Eder). Não há ground truth; comparam-se forma e concordância entre métodos, não ATE nem RPE.

## Resultados e Discussões (4000, usado 3303)

A análise de cor, independente do retreino, já está medida. Nos recortes rotulados, a regra de saturação heurística acertou 31,2% da distinção vivo × branqueado; com limiares otimizados subiu só a 58,4%. O índice V·(1−S) chegou a 82,1% e o brilho a 89,3%. Reabrir a faixa “pálido” caiu a acurácia para 80,2%, sem reduzir a detecção de branqueamento (201 de 241 nas duas configurações). Isso confirma Ferrera et al. (2019) no ponto em que o meio degrada a crominância: o véu dessatura a cena inteira e tira da saturação o poder de classe. O brilho sobrevive porque o carbonato exposto continua mais claro que o tecido. A mesma medição condena usar a cor para coral morto: 72% dos mortos caem em “saudável”. A literatura de monitoramento visual costuma tratar branqueamento como problema de cor; os dados mostram que morte e vida se separam por textura, não por média HSV. Daí a divisão de tarefas do pipeline.

[A PREENCHER — segmentação. Colar a tabela do evaluate.py no split de teste após o Colab: P, R, mAP50 e mAP50-95 da máscara para all, coral_vivo, coral_branqueado e coral_morto. Discutir se o recall de coral_morto saiu de 0,090 (treino curto em CPU, 30 épocas somadas) ou se permaneceu baixo. Se permanecer baixo com 130 épocas, o problema é ambiguidade visual, não orçamento de treino. No treino curto o mAP50 da máscara ainda subia ao fim da fase 2 (0,090 → 0,174), o que já indicava subtreino.]

Na odometria, o vídeo completo com keyframe a cada 3 quadros gerou 939 poses (ORB), 919 (SIFT) e 876 (KLT). Como cada passo tem comprimento 1, a “distância” percorrida coincide com o número de poses e não tem unidade física. Na vista de mapa (X–Z), SIFT e KLT seguem o mesmo quadrante (+X, −Z) e terminam a 81 unidades um do outro; o ORB segue o quadrante oposto (−X, −Z) e fecha a 256 unidades do SIFT. A distância média ao longo dos 876 primeiros passos é 54 entre SIFT e KLT, contra 171 (ORB–SIFT) e 205 (ORB–KLT). No eixo vertical o ORB quase não sobe (Y máximo 66), enquanto SIFT e KLT sobem a 161 e 194. Os três concordam no início e se separam com o tempo: deriva acumulada típica de VO quadro a quadro, sem bundle adjustment nem fechamento de loop.

Esse padrão conversa com a literatura e com a anotação de trabalho do próprio código (“ORB não é bom para ambientes subaquáticos”). Ferrera et al. (2019) já haviam preferido fluxo óptico a descritores binários sob turbidez; Nordfeldt-Fiol (2022) ganhou ao trocar o detector padrão por SIFT em fundo biológico. O ORB, rápido em terra, aqui é o método que mais mente. SIFT foi o mais estável na forma. KLT acompanha o SIFT quando o movimento é suave, mas descarta mais keyframes (876 contra 919), o que é coerente com a verificação ida e volta em trechos de turbidez.

Três limites impedem transformar o gráfico em erro métrico. Primeiro, não há pose de referência: AQUALOC existiria para isso, mas o teste usou vídeo de naufrágio, mais próximo da operação prevista do ROV. Segundo, a matriz K não foi calibrada, e a refração água–vidro–ar desloca o foco efetivo; `recoverPose` é sensível a K. Terceiro, a translação unitária distorce a forma se a velocidade variar entre keyframes. O resultado útil, neste estágio, é comparativo: em cena submersa com estrutura (costelas), SIFT e KLT são utilizáveis como esboço de trajetória; ORB não é.

## Conclusões (4000, usado 1950; o edital pede que esta aba seja breve)

O plano de trabalho foi cumprido no recorte realinhado: há um segmentador de corais com leitura de saúde e um módulo de odometria visual monocular comparando três métodos clássicos, ambos pensados para o ROV da APA Costa dos Corais.

Na saúde, o resultado mais sólido não é um índice, é uma negativa medida. Saturação não classifica branqueamento em imagem crua submersa; brilho classifica. Correção de cor, que a literatura de realce (Ancuti et al., 2018) recomenda para visualização, piorou a decisão quando o sinal útil era o brilho. Coral morto não se lê por cor. Essas três afirmações valem para a câmera, a profundidade e o pipeline usados na calibração; outro sítio exige repetir `gerar_calibracao.py` e `calibrate_health.py`. A faixa “pálido” é escolha operacional, não classe anotada.

[A PREENCHER — uma frase sobre o mAP final e o recall de coral_morto. Se o retreino elevar vivo e branqueado mas não morto, concluir que morte exige outra representação (textura/estrutura) ou mais dados da classe.]

Na odometria, o experimento confirma a literatura em condições controladas pelo próprio código: descritor binário (ORB) diverge; SIFT e KLT concordam na forma e acumulam deriva. Sem calibração da câmera, sem escala e sem ground truth, o módulo não é ainda um sensor de navegação. É um diagnóstico de qual front-end sobrevive à água e um ponto de partida. Os próximos passos objetivos são: calibrar K com tabuleiro submerso; recuperar escala com profundímetro ou objeto de tamanho conhecido; avaliar ATE/RPE em sequência pública (AQUALOC); e, se a deriva continuar inaceitável, passar de VO puro para um sistema com janela de otimização ou ORB-SLAM3, ciente de que o front-end ORB, sozinho, falhou aqui.

Recomenda-se manter as duas frentes desacopladas. A inferência conjunta (`pipeline_unificado.py`) já combina peixe-leão e coral no mesmo quadro; a odometria deve continuar como módulo de navegação, não como ramo da rede de segmentação.

## Referências (4000, usado 2117)

ANCUTI, C. O.; ANCUTI, C.; DE VLEESCHOUWER, C.; BEKAERT, P. Color balance and fusion for underwater image enhancement. IEEE Transactions on Image Processing, v. 27, n. 8, p. 379–393, 2018.

AZHMUKHAMEDOV, I. M.; TAMKOV, P. I.; SVISHCHEV, N. D.; RYBAKOV, A. V. Visual odometry in local underwater navigation problems. Journal of Physics: Conference Series, v. 2091, p. 012053, 2021.

CORALWATCH. Coral Health Chart. Disponível em: https://coralwatch.org. Acesso em: 24 ago. 2026.

FERRERA, M.; CREUZE, V.; MORAS, J.; TROUVÉ-PELOUX, P. AQUALOC: an underwater dataset for visual–inertial–pressure localization. The International Journal of Robotics Research, v. 38, n. 14, p. 1549–1559, 2019.

FERRERA, M.; MORAS, J.; TROUVÉ-PELOUX, P.; CREUZE, V. Real-time monocular visual odometry for turbid and dynamic underwater environments. Sensors, v. 19, n. 3, p. 687, 2019.

JOCHER, G. et al. Ultralytics YOLO. 2023. Disponível em: https://github.com/ultralytics/ultralytics. Acesso em: 24 ago. 2026.

LOWE, D. G. Distinctive image features from scale-invariant keypoints. International Journal of Computer Vision, v. 60, n. 2, p. 91–110, 2004.

LUCAS, B. D.; KANADE, T. An iterative image registration technique with an application to stereo vision. In: International Joint Conference on Artificial Intelligence, 1981.

NORDFELDT-FIOL, B. M. Improving visual odometry for challenging underwater environments and AUV navigation. Dissertação (Mestrado) — Universitat de les Illes Balears, 2022.

RUBLEE, E.; RABAUD, V.; KONOLIGE, K.; BRADSKI, G. ORB: an efficient alternative to SIFT or SURF. In: IEEE International Conference on Computer Vision, 2011.

SAUDER, J.; DOMAZETOSKI, V.; BANC-PRANDI, G.; PERNA, G.; MEIBOM, A.; TUIA, D. The Coralscapes dataset: semantic scene understanding in coral reefs. arXiv:2503.20000, 2025.

SCARAMUZZA, D.; FRAUNDORFER, F. Visual odometry: part I — the first 30 years and fundamentals. IEEE Robotics & Automation Magazine, v. 18, n. 4, p. 80–92, 2011.

ZHANG, J.; ILA, V.; KNEIP, L. Robust visual odometry in underwater environment. In: OCEANS 2018 MTS/IEEE Kobe Techno-Oceans, p. 1–9, 2018.
