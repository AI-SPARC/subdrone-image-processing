# subdrone-image-processing

Estudo de aplicações de análise de imagem para o contexto de um ROV
subaquático, no âmbito do projeto de monitoramento de longo prazo da APA Costa
dos Corais.

O repositório reúne três frentes independentes, que compartilham o mesmo tipo
de material de entrada: vídeo e imagem capturados pelo veículo em ambiente
recifal.

| pasta | frente |
|---|---|
| [`segmentation/corais/`](segmentation/corais/) | segmentação de corais e estimativa de saúde por cor |
| [`segmentation/peixe-leao/`](segmentation/peixe-leao/) | detecção do peixe-leão invasor por caixa delimitadora |
| [`segmentation/pipeline_unificado.py`](segmentation/pipeline_unificado.py) | inferência conjunta das duas redes sobre o mesmo frame |
| [`visual-odometry-sub/`](visual-odometry-sub/) | odometria visual monocular para estimar a trajetória do ROV |
| [`artigos/`](artigos/) | referências bibliográficas usadas no estudo |

Cada frente tem seu próprio README com metodologia, resultados medidos e
instruções de execução.

## Organização

Datasets, ambientes virtuais e pesos de modelos não são versionados. Os
READMEs de cada frente documentam a origem dos dados e os comandos necessários
para reproduzir o pipeline a partir do zero.

As detecções de peixe-leão e a segmentação de corais são resolvidas por dois
modelos separados, e não por um único modelo multitarefa. Os formatos de
anotação são diferentes (caixa contra polígono), os datasets são independentes,
e manter os modelos separados permite retreinar um sem afetar o outro. A
integração acontece na camada de inferência, em `pipeline_unificado.py`, que
recebe um frame e devolve as duas leituras em um relatório único.
