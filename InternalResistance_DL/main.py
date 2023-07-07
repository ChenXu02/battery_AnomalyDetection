import argparse
import pytorch_lightning as pl
import models
import tasks
import utils.data
from pytorch_lightning.callbacks import ModelCheckpoint





def get_model(args, dm):
    model = None
    if args.model_name == "GRU":
        model = models.GRU(input_dim=dm.inputdim, hidden_dim=args.hidden_dim)
    return model


def get_task(args, model, dm):
    task = getattr(tasks, args.settings.capitalize() + "ForecastTask")(
        model=model, feat_max_val=dm.feat_max_val,feat_max_val_label=dm.feat_max_val_label,feat_max_val2=dm.feat_max_val2,feat_max_val_label2=dm.feat_max_val_label2, **vars(args)
    )
    return task

def main_supervised(args):
    dm = utils.data.SpatioTemporalCSVDataModule(
        feat_path=DATA_PATHS[args.data]["feat"],feat_path2=DATA_PATHS[args.data]["feat2"], **vars(args)
    )
    model = get_model(args, dm)
    task = get_task(args, model, dm)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")
    trainer = pl.Trainer.from_argparse_args(args,max_epochs = 100,callbacks=[checkpoint_callback])
    if args.train:
        trainer.fit(task, dm)
        results = trainer.validate(datamodule=dm)
    else:
        trainer.test(task,dm,ckpt_path="lightning_logs/version_9/checkpoints/epoch=6-step=798.ckpt")
        results=0
    return results


def main(args):
    results = globals()["main_" + args.settings](args)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)

    parser.add_argument(
        "--data", type=str, help="The name of the dataset", default="SOC"
    )
    parser.add_argument(
        "--train", type=int, help="train/test", default=0
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="The name of the model for spatiotemporal prediction",
        choices=("GRU"),
        default="GRU",
    )
    parser.add_argument(
        "--settings",
        type=str,
        help="The type of settings, e.g. supervised learning",
        choices=("supervised",),
        default="supervised",
    )
    parser.add_argument("--log_path", type=str, default=None, help="Path to the output console log file")

    temp_args, _ = parser.parse_known_args()
    parser = getattr(utils.data, temp_args.settings.capitalize() + "DataModule").add_data_specific_arguments(parser)
    parser = getattr(models, temp_args.model_name).add_model_specific_arguments(parser)
    parser = getattr(tasks, temp_args.settings.capitalize() + "ForecastTask").add_task_specific_arguments(parser)
    args = parser.parse_args()
    if args.train==1:
        DATA_PATHS = {
            "SOC": {"feat": "../Data/SOCtrainData_BJJCf_same_train.npz","feat2": "../Data/SOCtrainData_BJJCf_same1_val.npz"},
            # "SOC": {"feat": "data/SOCtrainData_BJJCf_4500_test.npz"},
        }
    else:
        file_t = "../Data/SOCtrainData_BJJCf_same_val.npz"
        DATA_PATHS = {
            #"SOC": {"feat": "data/SOCtrainData_BJJCf_train.npz"},
            "SOC": {"feat": file_t,"feat2": file_t},
        }
    results = main(args)


