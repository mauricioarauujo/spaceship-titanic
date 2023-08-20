# Kaggle's Spaceship-Titanic Project with Kedro and MLflow.

Welcome to the Spaceship-Titanic project repository, where we explore and apply MLOps techniques using the Kedro and MLflow tools. In this project, I aim to enhance my experiences in developing, training, and deploying Machine Learning models, following best practices.

## About the Project

The Spaceship-Titanic project is an exciting journey to explore and analyze the famous Titanic dataset available on Kaggle. We use Kedro to structure our data science workflow and MLflow to track and manage our models. This allows us to achieve a more organized and reproducible development process.

While the primary focus of this project is the implementation of MLOps practices using Kedro and MLflow, it's gratifying to note that we've achieved a respectable position on the Kaggle leaderboard. Currently, we are ranked in the top 14% in the Titanic Kaggle competition with not as much effort (yet) put into exploratory analysis and feature engineering.

This achievement demonstrates that it is possible to balance the implementation of efficient development, training, and model monitoring processes with competitive performance in Machine Learning competitions.

In this project, we have applied MLOps practices, including:

- Utilizing Kedro to structure the data science workflow.
- Tracking metrics, experiments, and model versions with MLflow.
- Automated model "deployment" to ensure environment consistency.
- Continuous integration and continuous delivery (CI/CD) to automate model training and "deployment".

We are pleased to share our journey of learning and applying MLOps with you in this project.

## Directories

Our directories follow the Kedro structure:

- **src/**: Contains project modules and their pipelines.
- **conf/base/**: Shared project configurations.
- **conf/local/**: Environment-specific configurations (not pushed to git).
- **notebooks/**: Interactive notebooks for data analysis and visualization.
- **data/**: Stores raw, intermediate, and processed data.
- **mlruns/**: Created after the first run. Stores all information (logs, models, parameters, metrics) related to MLflow.

More info: [Kedro Architure Overview](https://docs.kedro.org/en/0.18.3/faq/architecture_overview.html#:~:text=Kedro%20project&text=The%20conf%2F%20directory%2C%20which%20contains,source%20code%20for%20your%20pipelines.)

## Workflow

![image](https://github.com/mauricioarauujo/spaceship-titanic/assets/58861384/b9beb62c-4afc-48ff-8a27-1107a0b8733e)

## Pipelines

### Pipeline Registry

Pipeline names are in [src/spaceship_titanic/pipeline_registry.py](src/spaceship_titanic/pipeline_registry.py).

- `__default__`: With just `kedro run`, the preprocessing, retraining, and submission data inference pipelines will run.
- `pp`: Preprocessing
- `train`: Preprocessing + retraining.
- `tune`: Preprocessing + tuning of different candidate models.
- `inference`: Takes the production model and performs inference on submission data.

### Pipelines Descriptions

- `pp`: Where the data is processed and features are generated. This pipeline is reused during inference.
  
- `train`: The production model is loaded and retrained with new data. The production model is replaced by the retrained model only when the retrained model's performance on the new test data is at least 3% better than the performance of the current untrained model. Such verification does not make sense for this project since the data is static.
  
- `tune`: Among a series of candidate models, each model is tuned with training data, and the model that performs best on test data is selected. Afterward, it is automatically verified if this model is superior to the current production model; if yes, this model is registered in staging. If there is no production model (for example, the first run), the candidate model is automatically saved if its performance is better than a given threshold.
  
- `inference`: In the submission competition's feature data, the same preprocessing and data treatments as the training data are applied. Upon these data, the production model is used to generate the submission csv file with IDs and predictions.

## Getting Started

1. Clone this repository: `git clone https://github.com/mauricioarauujo/spaceship-titanic.git`
2. Install the necessary dependencies: `make install` for runtime and `make install-dev` for development.
3. Explore the Kedro workflow in [src/](src/).
4. Monitor and manage your experiments in MLflow.
5. Run the default pipeline with the command: `kedro run`
6. To run individual pipelines, execute the command: `kedro run -p {pipeline_name}`. For example, `kedro run -p pp` will run the preprocessing pipeline. Usually, we will want to run the tuning pipeline (which may take some time) to deploy a good model in production and then run the inference pipeline.
7. For experiment/run visualizations and models saved with MLflow, run the command: `kedro mlflow ui`. Furthermore, through this interface, we can manually manage which registered models will be used in production.

   
## Contribution

Feel free to create pull requests or open issues to discuss improvements and new ideas.

## License

This project is licensed under the [MIT License](LICENSE).

---

Developed by Mauricio Araujo (https://github.com/mauricioarauujo)
