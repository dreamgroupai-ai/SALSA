# -------------------------
# Hydra entrypoint
# -------------------------
import hydra
from omegaconf import DictConfig

from src.salsa_trainer import TaskTrainer


@hydra.main(config_path="configs", config_name="task13a", version_base="1.3")
def main(config: DictConfig):
    trainer = TaskTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
