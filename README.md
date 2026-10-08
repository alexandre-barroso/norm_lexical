# Utilidade gramatical localizada de intervenções lexicais

Código de reprodução do estudo com DANTEStocks e Porttinari. O experimento substitui uma única forma por um alvo CorrectForm ou FullForm fornecido pela anotação, conserva o restante da entrada e compara as previsões UPOS focais e não focais. Não implementa um normalizador nem uma avaliação de compreensão ou preservação discursiva. **OBSERVAÇÃO: O arquivo CITATION.cff não encontra-se no repositório para fins de anonimato durante revisão por pares.**

## Ambiente e verificação

Python 3.12.3; dependências fixadas em `requirements.txt`. Instale-as em um ambiente virtual local. Não são necessários modelos NLTK baixados, Lean, Mathematica ou acesso aos materiais privados que inspiraram o projeto.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B verify.py
.venv/bin/python acquire_inputs.py inputs
PYTHONHASHSEED=0 .venv/bin/python -B pt_task_utility.py run
.venv/bin/python -B verify.py --results run
```

`verify.py` encerra com código diferente de zero se algum teste falhar. Sem `--results`, executa somente os treze testes sintéticos e declara que não reproduziu os resultados de corpus. A verificação completa compara os 2.352 pares de previsões, as dezesseis células de resumo e os oito registros de modelo/regime com os resultados originais. O SHA-256 dos eventos precisa coincidir exatamente. Tempo de execução e caminhos de arquivos não são alvos de igualdade numérica.

A execução preserva arquivos existentes e recusa sobrescrever resultados. Para uma nova execução, use uma pasta de saída nova: `UPOS_OUTPUT=outro_run`. `UPOS_INPUTS` permite indicar outra pasta com os quatro arquivos autenticados. Esses diretórios são criados localmente; nunca publique automaticamente os dados, modelos ou resultados por evento.

## Entradas e alcance

`inputs.json` contém as quatro URLs públicas fixadas por commit, nomes e hashes completos. O carregador aceita apenas treinamento e desenvolvimento; não lê a partição de teste. Os corpora são distribuídos separadamente pelos responsáveis, sob CC BY 4.0 e respectivos avisos. Os arquivos brutos não integram este repositório. A aquisição não executa código externo.

Porttinari: commit `87a07e1fb761d6d0a6e2a4d82b11b308344dabb9`. DANTEStocks: commit `4268e8e1b2c95708136e34f38b8e8af96de86dd2`. As convenções das versões anotadas, as exclusões por sobreposição, as sementes 7/23/47 e as cinco épocas são parte do objeto reproduzido.

`expected_results.json` contém resultados derivados, sem textos integrais de tweets. `export_provenance.json` distingue a implementação original e a adaptação de caminhos para reprodução. O novo registro de execução não se apresenta como o congelamento prospectivo original. O recurso já havia sido explorado anteriormente; esta é uma análise descritiva de desenvolvimento.

## Correspondência com o artigo

- `pt_task_utility.py`: leitura, elegibilidade, agrupamento, filtragem, treinamento, controle literal e intervenções pareadas; produz `run/results.json`, `run/events.jsonl` e parâmetros JSON.
- `verify.py`: treze verificações sintéticas e comparação com a execução documentada nos materiais do artigo.
- `expected_results.json`: tabelas de resultados, decomposição focal/não focal, cobertura lexical e concentrações.
- `inputs.json`: aquisição e autenticação, sem cópias dos dados.

As somas contextuais contam exposições repetidas em ensaios separados. Não representam normalização simultânea de todos os tweets, tokens independentes, significância populacional ou novos julgamentos humanos. As convenções UPOS e a tokenização da entrada original permanecem fixas.

Nenhum repositório remoto foi criado. A publicação e a escolha de licença para o código original dependem do autor; não se presume uma licença pública. A configuração de CI é fornecida, mas não há execução hospedada ou badge de sucesso alegado. A assistência efetivamente utilizada é declarada no manuscrito.

A figura científica editável é regenerada por `python plot_figure.py` após instalar `requirements-figures.txt`. O comando produz PDF, PNG (450 dpi), JPEG e os dados numéricos representados em `figures/`. A variável opcional `MANUSCRIPT_AUTHOR` define a autoria nos metadados de uma cópia identificada; o padrão é neutro.
