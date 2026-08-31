from .models import save_model
import os
import csv

class Callback:
    def __init__(self, *args, **kwargs):
        pass
    
    def callOnEndForward(self, trainer, model):
        pass

    def callOnBeginTrain(self, trainer, model):
        pass
    
    def callOnEndTrain(self, trainer, model):
        pass

    def callOnEndTest(self, trainer, model):
        pass
    

class LoggerCB(Callback):
    def __init__(self, file_path):
        super().__init__()
        self.path = file_path
        self.logged = {"epoch" : -1, "time" : 0, "train_loss" : -1, "test_loss" : -1}
        with open(self.path, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.logged.keys())
            writer.writeheader()

    def callOnEndTrain(self, trainer, model):
        self.logged.update({"epoch": trainer.metrics["epoch"]})
        self.logged.update({"time": trainer.metrics["epoch_time"]})
        self.logged.update({"train_loss": trainer.metrics["train_loss"]})
        self.logged.update({"test_loss": trainer.metrics.get("test_loss", -1)})
        trainer.log(f"Train loss after epoch {trainer.metrics['epoch']} : {trainer.metrics['train_loss']}")
        self._write_log()

    def callOnEndTest(self, trainer, model):
        trainer.log(f"Test loss after epoch {trainer.metrics['epoch']} : {trainer.metrics['test_loss']}")

    def _write_log(self):
        with open(self.path, "a") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.logged.keys())
            writer.writerow(self.logged)

class CheckpointCB(Callback):
    def __init__(self, when : dict):
        self.when = when

    def callOnEndTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if epoch in self.when:
            name = f"model_e{epoch}.pt"
            path = os.path.join(trainer.config.output_folder, name)
            save_model(model, path)

class UpdateHkrRegulCB(Callback):

    def __init__(self, when : dict):
        super().__init__()
        self.when = when

    def callOnBeginTrain(self, trainer, model):
        epoch = trainer.metrics["epoch"]
        if epoch in self.when:
            trainer.config.loss_regul = self.when[epoch]
            trainer.log("Updated loss regul weight to", self.when[epoch])
