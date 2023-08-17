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
2. Instale as dependências necessárias: `pip install -r requirements.txt`
3. Explore o fluxo de trabalho do Kedro em [src/](src/).
4. Acompanhe e gerencie seus experimentos no MLflow.
5. Execute o pipeline de treinamento com o comando: `kedro run`
6. [EXTRA] Para executar pipelines individualmente execute o comando: kedro run -p {nome_do_pipeline}. Os nomes dos pipelines estão em src/spaceship_titanic/pipeline_registry.py

## Fluxo de Trabalho

Nosso fluxo de trabalho segue a estrutura do Kedro:

- **src/**: Contém os módulos do projeto e seus pipelines.
- **conf/base/**: Configurações de projeto compartilhadas.
- **conf/local/**: Configurações específicas do ambiente local (não sobem para o git).
- **notebooks/**: Notebooks interativos para análise e visualização dos dados.
- **data/**: Armazena dados brutos, intermediários e processados.

## Contribuição

Adoramos contribuições! Sinta-se à vontade para criar pull requests ou abrir problemas para discutir melhorias e novas ideias.

## Licença

Este projeto está licenciado sob a [Licença MIT](LICENSE).

---

Desenvolvido por Mauricio Araujo (https://github.com/mauricioarauujo)
