#!/usr/bin/env python3
"""Recipe for fine-tuning a wav2vec model for the ST task (no transcriptions).

Author
 * Marcely Zanon Boito, 2022
"""

import sys
import os
import torch
import logging
from pathlib import Path
import torch
from hyperpyyaml import load_hyperpyyaml
from sacremoses import MosesDetokenizer


import speechbrain as sb
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.logger import get_logger

logger = logging.getLogger(__name__)

os.environ['WANDB__SERVICE_WAIT'] = '999999'

# Define training procedure
class ST(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig  # audio
        tokens_bos, _ = batch.tokens_bos  # translation

        # # wav2vec module
        # feats = self.modules.wav2vec2(wavs, wav_lens)

        # # dimensionality reduction
        # src = self.modules.enc(feats)

        # # transformer decoder
        # if self.distributed_launch:
        #     dec_out = self.modules.Transformer.module.forward_mt_decoder_only(
        #         src, tokens_bos, pad_idx=self.hparams.pad_index
        #     )
        # else:
        #     dec_out = self.modules.Transformer.forward_mt_decoder_only(
        #         src, tokens_bos, pad_idx=self.hparams.pad_index
        #     )

        # # logits and softmax
        # pred = self.modules.seq_lin(dec_out)
        # p_seq = self.hparams.log_softmax(pred)
        
        # compute features
        feats = self.hparams.compute_features(wavs) # (B, T, 80)
        current_epoch = self.hparams.epoch_counter.current
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        # Add feature augmentation if specified.
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "fea_augment"):
            feats, fea_lens = self.hparams.fea_augment(feats, wav_lens)
            tokens_bos = self.hparams.fea_augment.replicate_labels(tokens_bos)

        # forward modules
        src = self.modules.CNN(feats) # (B, L, 20, 32) -> (B, L, 640)

        enc_out, pred = self.modules.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index,
        )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # # compute outputs
        # hyps = None
        # if stage == sb.Stage.VALID:
        #     # the output of the encoder (enc) is used for valid search
        #     hyps, _, _, _ = self.hparams.valid_search(src.detach(), wav_lens)

        # elif stage == sb.Stage.TEST:
        #     hyps, _, _, _ = self.hparams.test_search(src.detach(), wav_lens)

        # return p_seq, wav_lens, hyps

        # Compute outputs
        hyps = None
        current_epoch = self.hparams.epoch_counter.current
        is_valid_search = (
            stage == sb.Stage.VALID
            and current_epoch % self.hparams.valid_search_interval == 0
        )
        is_test_search = stage == sb.Stage.TEST

        if any([is_valid_search, is_test_search]):
            # Note: For valid_search, for the sake of efficiency, we only perform beamsearch with
            # limited capacity and no LM to give user some idea of how the AM is doing

            # Decide searcher for inference: valid or test search
            if stage == sb.Stage.VALID:
                hyps, _, _, _ = self.hparams.valid_search(
                    enc_out.detach(), wav_lens
                )
            else:
                hyps, _, _, _ = self.hparams.test_search(
                    enc_out.detach(), wav_lens
                )

        return p_ctc, p_seq, wav_lens, hyps        

        

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given predictions and targets."""
        (p_ctc, p_seq, wav_lens, hyps, ) = predictions
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        # # st loss
        # loss = self.hparams.seq_cost(p_seq, tokens_eos, length=tokens_eos_lens)

        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "fea_augment"):
                tokens = self.hparams.fea_augment.replicate_labels(tokens)
                tokens_lens = self.hparams.fea_augment.replicate_labels(
                    tokens_lens
                )
                tokens_eos = self.hparams.fea_augment.replicate_labels(
                    tokens_eos
                )
                tokens_eos_lens = self.hparams.fea_augment.replicate_labels(
                    tokens_eos_lens
                )

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        ).sum()

        loss_ctc = self.hparams.ctc_cost(
            p_ctc, tokens, wav_lens, tokens_lens
        ).sum()

        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        # fr_detokenizer = MosesDetokenizer(lang=self.hparams.lang)

        # if stage != sb.Stage.TRAIN:
        #     predictions = [
        #         fr_detokenizer.detokenize(
        #             tokenizer.sp.decode_ids(utt_seq).split(" ")
        #         )
        #         for utt_seq in hyps
        #     ]
        #     detokenized_translation = [
        #         fr_detokenizer.detokenize(translation.split(" "))
        #         for translation in batch.trans
        #     ]
        #     # it needs to be a list of list due to the extend on the bleu implementation
        #     targets = [detokenized_translation]
        #     self.bleu_metric.append(ids, predictions, targets)
        #     # compute the accuracy of the one-step-forward prediction
        #     self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        # return loss

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                # Decode token terms to words
                predicted_words = [
                    tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
                ]
                target_words = [wrd.split(" ") for wrd in batch.wrd]
                self.bleu_metric.append(ids, predicted_words, target_words)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss

    # def init_optimizers(self):
    #     self.adam_optimizer = self.hparams.adam_opt_class(
    #         self.hparams.model.parameters()
    #     )

    #     self.optimizers_dict = {"model_optimizer": self.adam_optimizer}

    #     # Initializes the wav2vec2 optimizer if the model is not wav2vec2_frozen
    #     if not self.hparams.wav2vec2_frozen:
    #         self.wav2vec_optimizer = self.hparams.wav2vec_opt_class(
    #             self.modules.wav2vec2.parameters()
    #         )
    #         self.optimizers_dict["wav2vec_optimizer"] = self.wav2vec_optimizer

    # def freeze_optimizers(self, optimizers):
    #     """Freezes the wav2vec2 optimizer according to the warmup steps"""
    #     valid_optimizers = {}
    #     if not self.hparams.wav2vec2_frozen:
    #         valid_optimizers["wav2vec_optimizer"] = optimizers[
    #             "wav2vec_optimizer"
    #         ]
    #     valid_optimizers["model_optimizer"] = optimizers["model_optimizer"]
    #     return valid_optimizers

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
           ckpts, recoverable_name="model",
        )
        
        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()
        print("Loaded the average")

    def on_stage_start(self, stage, epoch):
        """Gets called when a stage (either training, validation, test) starts."""
        self.bleu_metric = self.hparams.bleu_computer()

        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.bleu_metric = self.hparams.bleu_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_loss
        else:  # valid or test
            # stage_stats = {"loss": stage_loss}
            # stage_stats["ACC"] = self.acc_metric.summarize()
            # stage_stats["BLEU"] = self.bleu_metric.summarize(field="BLEU")
            # stage_stats["BLEU_extensive"] = self.bleu_metric.summarize()
            # current_epoch = self.hparams.epoch_counter.current
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["BLEU"] = self.bleu_metric.summarize(field="BLEU")
                stage_stats["BLEU_extensive"] = self.bleu_metric.summarize()

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:
            # current_epoch = self.hparams.epoch_counter.current
            # old_lr_adam, new_lr_adam = self.hparams.lr_annealing_adam(
            #     stage_stats["BLEU"]
            # )
            # sb.nnet.schedulers.update_learning_rate(
            #     self.adam_optimizer, new_lr_adam
            # )

            # if not self.hparams.wav2vec2_frozen:
            #     (
            #         old_lr_wav2vec,
            #         new_lr_wav2vec,
            #     ) = self.hparams.lr_annealing_wav2vec(stage_stats["BLEU"])
            #     sb.nnet.schedulers.update_learning_rate(
            #         self.wav2vec_optimizer, new_lr_wav2vec
            #     )
            #     self.hparams.train_logger.log_stats(
            #         stats_meta={
            #             "epoch": current_epoch,
            #             "lr_adam": old_lr_adam,
            #             "lr_wav2vec": old_lr_wav2vec,
            #         },
            #         train_stats={"loss": self.train_stats},
            #         valid_stats=stage_stats,
            #     )
            # else:
            #     self.hparams.train_logger.log_stats(
            #         stats_meta={"epoch": current_epoch, "lr_adam": old_lr_adam},
            #         train_stats={"loss": self.train_stats},
            #         valid_stats=stage_stats,
            #     )

            # # create checkpoint
            # meta = {"BLEU": stage_stats["BLEU"], "epoch": current_epoch}
            # name = "checkpoint_epoch" + str(current_epoch)

            # self.checkpointer.save_and_keep_only(
            #     meta=meta, name=name, num_to_keep=10, max_keys=["BLEU"]
            # )
            lr = self.hparams.noam_annealing.current_lr
            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )

            self.checkpointer.save_and_keep_only(test_data,
                meta={"BLEU": stage_stats["BLEU"], "epoch": epoch},
                max_keys=["BLEU"],
                num_to_keep=self.hparams.avg_checkpoints,
            )

        elif stage == sb.Stage.TEST:
            # self.hparams.train_logger.log_stats(
            #     stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
            #     test_stats=stage_stats,
            # )
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_bleu_file, "w") as w:
                    self.bleu_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"BLEU": 1.1, "epoch": epoch},
                max_keys=["BLEU"],
                num_to_keep=1,
            )

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """At the end of the optimizer step, apply noam annealing."""
        if should_step:
            self.hparams.noam_annealing(self.optimizer)


# Define custom data procedure
def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """


    data_folder = hparams["data_folder"]

    @sb.utils.data_pipeline.takes("path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        """Load the audio signal. This is done on the CPU in the `collate_fn`."""
        sig = sb.dataio.dataio.read_audio(wav)
        return sig


    @sb.utils.data_pipeline.takes("path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline_train(wav):
        # Speed Perturb is done here so it is multi-threaded with the
        # workers of the dataloader (faster).
        if "speed_perturb" in hparams:
            sig = sb.dataio.dataio.read_audio(wav)
            sig = hparams["speed_perturb"](sig.unsqueeze(0)).squeeze(0)
        else:
            sig = sb.dataio.dataio.read_audio(wav)
        return sig

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("trans")
    @sb.utils.data_pipeline.provides(
        "trans", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens


    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["train_json"], replacements={"data_root": data_folder},
        dynamic_items=[audio_pipeline_train, text_pipeline],
            output_keys=[
                "id",
                "sig",
                "trans",
                "tokens_list",
                "tokens_bos",
                "tokens_eos", 
                "tokens"
            ],
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["valid_json"], replacements={"data_root": data_folder},
        dynamic_items=[audio_pipeline_train, text_pipeline],
            output_keys=[
                "id",
                "sig",
                "wrd",
                "tokens_list",
                "tokens_bos",
                "tokens_eos", 
                "tokens"
            ],
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        valid_data = valid_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(sort_key="duration", reverse=True)
        valid_data = valid_data.filtered_sorted(sort_key="duration", reverse=True)
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["train_dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )
   
    

    # # test is separate
    # test_datasets = {}
    # for json_file in hparams["test_json"]:
    #     name = Path(json_file).stem
    #     print("name:", name)
    #     test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_json(
    #         json_path=json_file, replacements={"data_root": data_folder}
    #     )
    #     test_datasets[name] = test_datasets[name].filtered_sorted(
    #         sort_key="duration"
    #     )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(
        json_path=hparams["test_json"], replacements={"data_root": data_folder},
        dynamic_items=[audio_pipeline_train, text_pipeline],
            output_keys=[
                "id",
                "sig",
                "wrd",
                "tokens_list",
                "tokens_bos",
                "tokens_eos", 
                "tokens"
            ],
    )
    # test_data = test_data.filtered_sorted(sort_key="duration")

    # datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]
    # valtest_datasets = [valid_data] + [i for k, i in test_datasets.items()]

    # datasets = [train_data, valid_data, test_data]
    # valtest_datasets = [valid_data, test_data] 

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # Define audio pipeline. In this case, we simply read the path contained
    # in the variable wav with the audio reader.
   
    

    # sb.dataio.dataset.add_dynamic_item(valtest_datasets, audio_pipeline)

    # @sb.utils.data_pipeline.takes("wav")
    # @sb.utils.data_pipeline.provides("sig")
    # def sp_audio_pipeline_train(wav):
    #     """Load the audio signal. This is done on the CPU in the `collate_fn`."""
    #     sig = sb.dataio.dataio.read_audio(wav)
    #     sig = sig.unsqueeze(0)
    #     sig = hparams["speed_perturb"](sig)
    #     sig = sig.squeeze(0)
    #     return sig

    # sb.dataio.dataset.add_dynamic_item([train_data], audio_pipeline_train)

    # # Define text processing pipeline. We start from the raw text and then
    # # encode it using the tokenizer. The tokens with BOS are used for feeding
    # # decoder during training, the tokens with EOS for computing the cost function.
    # @sb.utils.data_pipeline.takes("trans")
    # @sb.utils.data_pipeline.provides(
    #     "trans", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    # )
    # def text_pipeline(translation):
    #     """Processes the transcriptions to generate proper labels"""
    #     yield translation
    #     tokens_list = tokenizer.sp.encode_as_ids(translation)
    #     yield tokens_list
    #     tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
    #     yield tokens_bos
    #     tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
    #     yield tokens_eos

    
    # sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    # sb.dataio.dataset.set_output_keys(
    #     datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    # )



    # data_folder = hparams["data_folder"]

    # # 1. train tokenizer on the data
    # tokenizer = SentencePiece(
    #     model_dir=hparams["save_folder"],
    #     vocab_size=hparams["vocab_size"],
    #     annotation_train=hparams["annotation_train"],
    #     annotation_read="trans",
    #     annotation_format="json",
    #     model_type="unigram",
    #     bos_id=hparams["bos_index"],
    #     eos_id=hparams["eos_index"],
    # )

    # # 2. load data and tokenize with trained tokenizer
    # datasets = {}
    # for dataset in ["train", "valid"]:
    #     json_path = hparams[f"annotation_{dataset}"]

    #     is_use_sp = dataset == "train" and "speed_perturb" in hparams
    #     audio_pipeline_func = sp_audio_pipeline if is_use_sp else audio_pipeline

    #     datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
    #         json_path=json_path,
    #         replacements={"data_root": data_folder},
    #         dynamic_items=[audio_pipeline_func, reference_text_pipeline],
    #         output_keys=[
    #             "id",
    #             "sig",
    #             "duration",
    #             "trans",
    #             "tokens_list",
    #             "tokens_bos",
    #             "tokens_eos",
    #         ],
    #     )

    # for dataset in ["valid", "test"]:
    #     json_path = hparams[f"annotation_{dataset}"]
    #     datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
    #         json_path=json_path,
    #         replacements={"data_root": data_folder},
    #         dynamic_items=[audio_pipeline, reference_text_pipeline],
    #         output_keys=[
    #             "id",
    #             "sig",
    #             "duration",
    #             "trans",
    #             "tokens_list",
    #             "tokens_bos",
    #             "tokens_eos",
    #         ],
    #     )

    # # Sorting training data with ascending order makes the code  much
    # # faster  because we minimize zero-padding. In most of the cases, this
    # # does not harm the performance.
    # if hparams["sorting"] == "ascending":
    #     if hparams["debug"]:
    #         datasets["train"] = datasets["train"].filtered_sorted(
    #             key_min_value={"duration": hparams["sorting_min_duration"]},
    #             key_max_value={"duration": hparams["sorting_max_duration"]},
    #             sort_key="duration",
    #             reverse=True,
    #         )
    #         datasets["valid"] = datasets["valid"].filtered_sorted(
    #             key_min_value={"duration": hparams["sorting_min_duration"]},
    #             key_max_value={"duration": hparams["sorting_max_duration"]},
    #             sort_key="duration",
    #             reverse=True,
    #         )
    #     else:
    #         datasets["train"] = datasets["train"].filtered_sorted(
    #             sort_key="duration"
    #         )
    #         datasets["valid"] = datasets["valid"].filtered_sorted(
    #             sort_key="duration"
    #         )

    #     hparams["dataloader_options"]["shuffle"] = False
    #     hparams["dataloader_options"]["shuffle"] = False
    # elif hparams["sorting"] == "descending":
    #     # use smaller dataset to debug the model
    #     if hparams["debug"]:
    #         datasets["train"] = datasets["train"].filtered_sorted(
    #             key_min_value={"duration": hparams["sorting_min_duration"]},
    #             key_max_value={"duration": hparams["sorting_max_duration"]},
    #             sort_key="duration",
    #             reverse=True,
    #         )
    #         datasets["valid"] = datasets["valid"].filtered_sorted(
    #             key_min_value={"duration": hparams["sorting_min_duration"]},
    #             key_max_value={"duration": hparams["sorting_max_duration"]},
    #             sort_key="duration",
    #             reverse=True,
    #         )
    #     else:
    #         datasets["train"] = datasets["train"].filtered_sorted(
    #             sort_key="duration", reverse=True
    #         )
    #         datasets["valid"] = datasets["valid"].filtered_sorted(
    #             sort_key="duration", reverse=True
    #         )

    #     hparams["dataloader_options"]["shuffle"] = False
    #     hparams["dataloader_options"]["shuffle"] = False
    # elif hparams["sorting"] == "random":
    #     # use smaller dataset to debug the model
    #     if hparams["debug"]:
    #         datasets["train"] = datasets["train"].filtered_sorted(
    #             key_min_value={"duration": hparams["sorting_debug_duration"]},
    #             key_max_value={"duration": hparams["sorting_max_duration"]},
    #             sort_key="duration",
    #         )
    #         datasets["valid"] = datasets["valid"].filtered_sorted(
    #             key_min_value={"duration": hparams["sorting_min_duration"]},
    #             key_max_value={"duration": hparams["sorting_max_duration"]},
    #             sort_key="duration",
    #         )

    #     hparams["dataloader_options"]["shuffle"] = True
    # else:
    #     raise NotImplementedError(
    #         "sorting must be random, ascending or descending"
    #     )

    # return datasets, tokenizer

    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler  # noqa

        dynamic_hparams_train = hparams["dynamic_batch_sampler_train"]
        dynamic_hparams_valid = hparams["dynamic_batch_sampler_valid"]

        # print(dynamic_hparams_train)

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_train,
        )
        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x["duration"],
            **dynamic_hparams_valid,
        )

    return (
        train_data,
        valid_data,
        test_data,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )


if __name__ == "__main__":
    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    print("run_opts:", run_opts)
    print("overrides:", overrides)
    print("hparams_file:", hparams_file)
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # creates a logger
    logger = get_logger(__name__)

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation
    import prepare_iwslt22
    # import prepare_indicst

    if not hparams["skip_prep"]:
        run_on_main(
            prepare_iwslt22.data_proc,
            kwargs={
                "dataset_folder": hparams["root_data_folder"],
                "output_folder": hparams["data_folder"],
            },
        )

    # here we create the datasets objects as well as tokenization and encoding
    (
        train_data,
        valid_data,
        test_data,
        tokenizer,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams)

     # We download the pretrained LM from HuggingFace (or elsewhere depending on
    # the path given in the YAML file). The tokenizer is loaded at the same time.
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected()

    # Init wandb
    if hparams['use_wandb']:
        hparams['train_logger'] = hparams['wandb_logger']()
        
    if hparams['no_lm']:
        print('Evaluate without LM.')
        hparams['test_search'] = hparams['valid_search']
        hparams["output_bleu_folder"] = os.path.join(hparams["output_bleu_folder"], 'no_lm')

    # Create main experiment class
    st_brain = ST(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    
    # # # Load datasets for training, valid, and test, trains and applies tokenizer
    # # datasets, tokenizer = dataio_prepare(hparams)

    # # Before training, we drop some of the wav2vec 2.0 Transformer Encoder layers
    # st_brain.modules.wav2vec2.model.encoder.layers = (
    #     st_brain.modules.wav2vec2.model.encoder.layers[
    #         : hparams["keep_n_layers"]
    #     ]
    # )

    # # Training
    # st_brain.fit(
    #     st_brain.hparams.epoch_counter,
    #     datasets["train"],
    #     datasets["valid"],
    #     train_loader_kwargs=hparams["dataloader_options"],
    #     valid_loader_kwargs=hparams["test_dataloader_options"],
    # )

    # # Test
    # for dataset in ["valid", "test"]:
    #     st_brain.evaluate(
    #         datasets[dataset],
    #         test_loader_kwargs=hparams["test_dataloader_options"],
    #     )

    # adding objects to trainer:
    st_brain.tokenizer = hparams["tokenizer"]
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]

    if train_bsampler is not None:
        collate_fn = None
        if "collate_fn" in train_dataloader_opts:
            collate_fn = train_dataloader_opts["collate_fn"]

        train_dataloader_opts = {
            "batch_sampler": train_bsampler,
            "num_workers": hparams["num_workers"],
        }

        if collate_fn is not None:
            train_dataloader_opts["collate_fn"] = collate_fn

    if valid_bsampler is not None:
        collate_fn = None
        if "collate_fn" in valid_dataloader_opts:
            collate_fn = valid_dataloader_opts["collate_fn"]

        valid_dataloader_opts = {"batch_sampler": valid_bsampler}

        if collate_fn is not None:
            valid_dataloader_opts["collate_fn"] = collate_fn

    if not hparams['skip_train']:
        # Training
        st_brain.fit(
            st_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=train_dataloader_opts,
            valid_loader_kwargs=valid_dataloader_opts,
        )

    # Testing
    if not os.path.exists(hparams["output_bleu_folder"]):
        os.makedirs(hparams["output_bleu_folder"])

    for k in test_data.keys():  # keys are test_clean, test_other etc
        st_brain.hparams.test_bleu_file = os.path.join(
            hparams["output_bleu_folder"], f"bleu_{k}.txt"
        )
        st_brain.evaluate(
            test_data[k],
            max_key="BLEU",
            test_loader_kwargs=hparams["test_dataloader_opts"],
        )