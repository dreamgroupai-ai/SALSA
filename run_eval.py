# -------------------------
# Hydra entrypoint
# -------------------------
import hydra
from omegaconf import DictConfig
import json, os
from src.salsa_trainer import TaskTrainer


@hydra.main(config_path="configs", config_name="agnews", version_base="1.3")
def main(config: DictConfig):
    trainer = TaskTrainer(config)
    results = trainer.evaluate(trainer.test_dataset)
    json_results = json.dumps(results, indent=4)
    print(f"Evaluation results:\n {json_results}")
    with open(f"eval_results_{config.model.id.replace('/', '_')}.json", "w") as f:
        f.write(json_results)


if __name__ == "__main__":
    main()
