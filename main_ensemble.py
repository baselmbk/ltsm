from src.utils.logger import get_logger
from src.training.ensemble_trainer import EnsembleTrainer
from src.utils.config import load_config


def main():
    logger = get_logger("ensemble_main")
    logger.info("Starting the ensemble cryptocurrency prediction process...")

    config = load_config('ensemble_config')
    data_config = load_config('data_config')

    trainer = EnsembleTrainer(config, data_config, logger)
    trainer.run()

    logger.info("Ensemble cryptocurrency prediction process completed.")


if __name__ == "__main__":
    main()