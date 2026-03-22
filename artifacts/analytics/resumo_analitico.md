# Questões analíticas

Documento com respostas, número-chave, leitura gerencial e ação recomendada.

## Pergunta 1 - Adequacao do nivel (IAN)

**Resposta curta:** A taxa de alunos em risco caiu de 69.9% em 2022 para 46.2% em 2024.

**Número-chave:** `46.2%`

**Leitura gerencial:** O programa melhora a adequacao de nivel ao longo do tempo, mas ainda termina 2024 com uma base relevante em risco.

**Ação recomendada:** Manter trilhas preventivas para alunos em defasagem moderada antes que migrem para risco severo.

**Figura:** `artifacts/analytics/01_ian_perfil_evolucao.png`

## Pergunta 2 - Desempenho academico (IDA)

**Resposta curta:** O IDA medio saiu de 6.09 para 6.66 no periodo, com leve acomodacao em 2024 frente a 2023.

**Número-chave:** `+0.26`

**Leitura gerencial:** O ganho academico existe, mas nao e linear em todos os ciclos.

**Ação recomendada:** Direcionar reforco academico aos ciclos em que o IDA desacelera para evitar acomodacao do aprendizado.

**Figura:** `artifacts/analytics/02_ida_evolucao.png`

## Pergunta 3 - Engajamento nas atividades (IEG)

**Resposta curta:** O engajamento tem relacao positiva com o desempenho e o ponto de virada: corr(IEG, IDA)=0.54 e corr(IEG, IPV)=0.56.

**Número-chave:** `0.54 / 0.56`

**Leitura gerencial:** Engajamento nao e so participacao: ele antecipa resultado academico e virada positiva.

**Ação recomendada:** Usar quedas de IEG como gatilho operacional para intervencao rapida por monitoria e relacionamento.

**Figura:** `artifacts/analytics/03_ieg_vs_ida_ipv.png`

## Pergunta 4 - Autoavaliacao (IAA)

**Resposta curta:** A autoavaliacao e parcialmente coerente com o desempenho real, mas a correlacao e fraca: corr(IAA, IDA)=0.12 e corr(IAA, IEG)=0.13.

**Número-chave:** `0.12`

**Leitura gerencial:** A percepcao do aluno ajuda, mas sozinha nao explica o que realmente acontece no desempenho.

**Ação recomendada:** Combinar autoavaliacao com sinais objetivos de engajamento e desempenho antes de decidir uma intervencao.

**Figura:** `artifacts/analytics/04_iaa_coerencia.png`

## Pergunta 5 - Aspectos psicossociais (IPS)

**Resposta curta:** Os padroes psicossociais antecedem mudancas de risco: a media de IPS difere em 0.60 pontos entre quem entra ou permanece em risco e quem segue fora de risco.

**Número-chave:** `0.60`

**Leitura gerencial:** O bloco psicossocial aparece antes da piora academica e pode ser usado como alerta preventivo.

**Ação recomendada:** Priorizar suporte psicossocial e contato com familia nos perfis com pior IPS mesmo antes da queda do INDE.

**Figura:** `artifacts/analytics/05_ips_risco_proximo.png`

## Pergunta 6 - Aspectos psicopedagogicos (IPP)

**Resposta curta:** O IPP acompanha a defasagem: a diferenca entre o maior e o menor valor medio de IPP e 0.67.

**Número-chave:** `0.67`

**Leitura gerencial:** As avaliacoes psicopedagogicas reforcam o diagnostico de defasagem e ajudam a qualificar a priorizacao.

**Ação recomendada:** Usar IPP para diferenciar alunos com defasagem estrutural de casos mais pontuais de oscilacao academica.

**Figura:** `artifacts/analytics/06_ipp_vs_ian.png`

## Pergunta 7 - Ponto de virada (IPV)

**Resposta curta:** O IPV e mais influenciado por ipp, com correlacao de 0.61.

**Número-chave:** `0.61`

**Leitura gerencial:** A virada positiva depende de um conjunto de sinais academicos e comportamentais, nao de um unico evento.

**Ação recomendada:** Tratar o IPV como indicador de combinacao de sinais e nao apenas como resultado final do ano.

**Figura:** `artifacts/analytics/07_ipv_drivers.png`

## Pergunta 8 - Multidimensionalidade dos indicadores

**Resposta curta:** A combinacao mais alinhada ao desempenho global atual e ida, com correlacao de 0.79 com o INDE.

**Número-chave:** `0.79`

**Leitura gerencial:** O INDE responde a um ecossistema de fatores; olhar um indicador isolado reduz a capacidade de agir cedo.

**Ação recomendada:** Adotar paines por aluno com visao combinada de engajamento, academico, psicossocial e psicopedagogico.

**Figura:** `artifacts/analytics/08_multidimensionalidade_heatmap.png`

## Pergunta 9 - Previsao de risco com Machine Learning

**Resposta curta:** No holdout 2023->2024, o modelo regressao_logistica atingiu recall=0.58, precision=0.50 e F2=0.56.

**Número-chave:** `F2=0.56`

**Leitura gerencial:** O modelo consegue antecipar parte relevante dos casos de risco usando apenas indicadores disponiveis antes do proximo ciclo.

**Ação recomendada:** Usar a previsao como fila de triagem, sempre combinada com leitura pedagogica e psicossocial da equipe.

**Figura:** `artifacts/analytics/09_modelo_holdout.png`

## Pergunta 10 - Efetividade do programa

**Resposta curta:** Em 2024, o ciclo com melhor desempenho medio foi Topazio, reforcando a efetividade heterogenea do programa.

**Número-chave:** `Topazio`

**Leitura gerencial:** O impacto do programa existe, mas nao acontece no mesmo ritmo em todos os ciclos.

**Ação recomendada:** Customizar metas e cadencia de acompanhamento por ciclo em vez de aplicar a mesma estrategia para toda a base.

**Figura:** `artifacts/analytics/10_efetividade_programa.png`

## Pergunta 11 - Tempo de programa e risco futuro

**Resposta curta:** O risco de proximo ano varia 20.5% entre as faixas de tempo de programa, indicando necessidade de acompanhamento segmentado.

**Número-chave:** `20.5%`

**Leitura gerencial:** Tempo de casa altera o padrao de risco, portanto a mesma acao nao serve para veteranos e ingressantes.

**Ação recomendada:** Segmentar acompanhamento por maturidade no programa, com trilhas especificas para ingressantes e alunos antigos.

**Figura:** `artifacts/analytics/11_insight_extra_tempo_programa.png`

# Análises complementares

## Transicoes de risco ano a ano

**Insight:** 10.5% entram em risco no ciclo seguinte, enquanto 20.2% conseguem se recuperar.

**Número-chave:** `10.5%`

**Leitura gerencial:** A base exige duas estrategias em paralelo: prevenir novas entradas em risco e recuperar quem ja esta vulneravel.

**Ação recomendada:** Criar uma fila operacional separando alunos que entram em risco dos que permanecem em risco para intervencoes diferentes.

**Figura:** `artifacts/analytics/12_transicoes_risco.png`

## Queda recente do INDE como alerta preventivo

**Insight:** A diferenca entre a pior e a melhor faixa de variacao do INDE e de 14.1% na taxa de risco do proximo ano.

**Número-chave:** `14.1%`

**Leitura gerencial:** A piora recente do INDE funciona como sinal precoce e de facil operacionalizacao para a equipe.

**Ação recomendada:** Acionar revisao de caso sempre que houver queda relevante de INDE, mesmo antes da consolidacao do risco.

**Figura:** `artifacts/analytics/13_delta_inde_alerta.png`

## Cobertura de avaliacoes como sinal operacional

**Insight:** A taxa de risco varia 31.8% entre alunos com baixa e alta cobertura de avaliacoes, sugerindo que monitoramento insuficiente caminha junto com maior vulnerabilidade.

**Número-chave:** `31.8%`

**Leitura gerencial:** Menor cobertura avaliativa deixa a equipe mais cega e coincide com maior risco futuro.

**Ação recomendada:** Garantir nivel minimo de avaliacoes por aluno para reduzir pontos cegos na gestao pedagogica.

**Figura:** `artifacts/analytics/14_cobertura_avaliacoes.png`
