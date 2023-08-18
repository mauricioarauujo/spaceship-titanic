# Spaceship-Titanic do Kaggle com Kedro e MLflow.

Bem-vindo ao repositório do projeto Spaceship-Titanic, onde exploramos e aplicamos técnicas de MLOps utilizando as ferramentas Kedro e MLflow. Neste projeto eu tenho como objetivo aprimorar minhas experiências de desenvolvimento, treinamento e implantação de modelos de Machine Learning, seguindo as melhores práticas.

## Sobre o Projeto

O projeto Spaceship-Titanic é uma jornada emocionante para explorar e analisar o famoso conjunto de dados do Titanic disponível no Kaggle. Usamos Kedro para estruturar nosso fluxo de trabalho de ciência de dados e MLflow para rastrear e gerenciar nossos modelos. Isso nos permite alcançar um processo de desenvolvimento mais organizado e reproduzível.

## Características Principais

- Estruturação do projeto usando Kedro para uma organização escalável e modular.
- Utilização do MLflow para rastrear métricas, experimentos e versões de modelo.
- Implementação de um pipeline de pré-processamento de dados para garantir consistência.
- Treinamento de modelos de Machine Learning com acompanhamento de métricas e parâmetros.
- Implantação de modelos de forma automatizada, garantindo a consistência do ambiente.

## Como Começar

1. Clone este repositório: `git clone https://github.com/mauricioarauujo/spaceship-titanic.git`
2. Instale as dependências necessárias: `make install` para rodagem e `make install-dev` para desenvolvimento.
3. Explore o fluxo de trabalho do Kedro em [src/](src/).
4. Acompanhe e gerencie seus experimentos no MLflow.
5. Execute o pipeline de treinamento com o comando: `kedro run`
6. Para visualizações de experimentos/rodagens e modelos salvos com mlflow execute o comando: `kedro mlflow ui`
7. [EXTRA] Para executar pipelines individualmente execute o comando: `kedro run -p {nome_do_pipeline}`. Os nomes dos pipelines estão em [src/spaceship_titanic/pipeline_registry.py](src/spaceship_titanic/pipeline_registry.py) (por exemplo, `kedro run -p pp` rodará o pipeline de preprocessamento).

## Fluxo de Trabalho

Nosso fluxo de trabalho segue a estrutura do Kedro:

- **src/**: Contém os módulos do projeto e seus pipelines.
- **conf/base/**: Configurações de projeto compartilhadas.
- **conf/local/**: Configurações específicas do ambiente local (não sobem para o git).
- **notebooks/**: Notebooks interativos para análise e visualização dos dados.
- **data/**: Armazena dados brutos, intermediários e processados.
- **mlruns/**: É criada após primeira rodagem. Armazenas todas as informações (logs, modelos, parametros, metricas) referentes ao mlflow.

## Pipeline Registry

- `__default__`: com apenas `kedro run`, será rodado o pipeline de preprocessamento, re-treino e inferencia nos dados de submissão.
- `pp`: Pre-processamento
- `train`: Pre-processamento + re-treino.
- `tune`: Pre-processamento + tunagem de diferentes modelos candidatos.
- `inference`: Pega o modelo em produção e faz a inferência nos dados de submissão.

## Pipelines

- `pp`: Onde é tratado o dado e gerado as features. Esse pipeline é reaproveitado no momento de inferência.
  
- `train`: É carregado e re-treinado o modelo em produção com os novos dados. O modelo em produção só é substituído pelo re-treinado quando a perfomance do modelo re-treinado for pelo menos 3% superior a performance do modelo vigente não re-treinado nos novos dados de teste. Tal verificação não faz sentido para o projeto em questão já que os dados são estáticos.
  
- `tune`: Dentre uma série de modelos candidatos é tunado cada modelo nos dados de treino e é pego o que melhor performa nos dados de teste. Após isso, é verificado automaticamente se este modelo é superior ao modelo em produção atual, se sim, tal modelo é registrado em staging. Se não houver modelo em produção (primeira rodagem, por exemplo), o modelo candidato é salvo automaticamente se sua performance for superior a um dado threshold.
  
- `inference`: É utilizado o pipeline de pre-processamento para gerar as mesmas features e o dado de entrada estar nos mesmos formatos dos dados que foram usados nos treinos. Com o modelo em produção, é gerado o arquivo de submissão para a competição no formato csv. 

## Contribuição

Adoramos contribuições! Sinta-se à vontade para criar pull requests ou abrir problemas para discutir melhorias e novas ideias. 

## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).

---

Desenvolvido por Mauricio Araujo (https://github.com/mauricioarauujo)
