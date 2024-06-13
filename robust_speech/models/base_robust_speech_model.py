
import logging
import os
import sys

import speechbrain as sb
import torch
import torch.nn as nn
import string
from transformers import PreTrainedModel, WhisperProcessor, Wav2Vec2ForCTC
import librosa

import robust_speech as rs
from robust_speech.adversarial.brain import AdvASRBrain
from typing import Optional, Union

logger = logging.getLogger(__name__)


# Define training procedure
class BaseASR(AdvASRBrain):
    """
    Base ASR model for running evaluations with Robust Speech. Some common boilerplate code is defined here.
    """
    def eval_forward(self,
                     feats=None,
                     feat_lens=None,
                     tokens=None,
                     tok_lens=None,
                     options=None,
                     loss_options=None):
        """
        Forward pass for evaluation
        Inputs:
            feats: torch.Tensor - feature input
            feat_lens: torch.Tensor - length of the features
            tokens: torch.Tensor - tokenized input
            tok_lens: torch.Tensor - length of the tokens
            options: dict - options for the forward pass
            loss_options: dict - options for the loss computation
        Returns:
            loss: torch.Tensor - loss value
            predictions: List[str] - predicted transcripts
        """
        raise NotImplementedError("eval_forward needs to be overridden in the derived class")
    
    def train_attack_forward(self,
                     feats=None,
                     feat_lens=None,
                     tokens=None,
                     tok_lens=None,
                     options=None,
                     loss_options=None):
        """
        Forward pass for training and attack
        Inputs:
            feats: torch.Tensor - feature input
            feat_lens: torch.Tensor - length of the features
            tokens: torch.Tensor - tokenized input
            tok_lens: torch.Tensor - length of the tokens
            options: dict - options for the forward pass
            loss_options: dict - options for the loss computation
        Returns:
            loss: torch.Tensor - loss value
            predictions: List[str] - predicted transcripts
        """
        raise NotImplementedError("train_attack_forward needs to be overridden in the derived class")
    
    def text_to_tokens(self, batch):
        """
        Tokenize the text input
        Inputs:
            batch: object - batch object
        Returns:
            tokens: torch.Tensor - tokenized input
            token_lens: torch.Tensor - length of the tokens
        """
        raise NotImplementedError("text_to_tokens needs to be overridden in the derived class")
    
    def wav_to_feats(self, batch):
        """
        Compute features from the waveform input
        Inputs:
            batch: object - batch object
        Returns:
            feats: torch.Tensor - feature input
            feat_lens: torch.Tensor - length of the features
        """
        raise NotImplementedError("wav_to_feats needs to be overridden in the derived class")


    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        if not stage == sb.Stage.TRAIN:
            self.modules = self.modules.eval()
        if not stage == rs.Stage.ATTACK:
            batch = batch.to(self.device)
        
        tokens, token_lens = self.text_to_tokens(batch)
        
        options = {}
        loss_options = {}
        if options.get("fp16", False):
            self.modules.to(torch.float16)
        
        # Use the feature extractor (if one is defined to compute features)
        feats, feat_lens = self.wav_to_feats(batch)

        # Forward pass
        if stage != sb.Stage.TRAIN and stage != rs.Stage.ATTACK:
            # Decode token terms to words
            loss, predictions = self.eval_forward(
                feats=feats,
                feat_lens=feat_lens,
                tokens=tokens,
                tok_lens=token_lens,
                options=options,
                loss_options=loss_options
            )
        else:
            loss, predictions = self.train_attack_forward(
                feats=feats,
                feat_lens=feat_lens,
                tokens=tokens,
                tok_lens=token_lens,
                options=options,
                loss_options=loss_options
            )
        return loss, predictions, stage

    def compute_objectives(
        self, predictions, batch, stage, adv=False, targeted=False, reduction="mean"
    ):
        """Computes the loss (CTC+NLL) given predictions and targets."""

        loss, predicted, save_stage = predictions
        ids = batch.id
        
        if stage != sb.Stage.TRAIN and stage != rs.Stage.ATTACK:
            # Compute case and punctuation insensitive WER/CER.
            predicted = [wrd.upper().translate(str.maketrans(
                '', '', string.punctuation)) for wrd in predicted]
            predicted_words = [wrd.split(" ") for wrd in predicted]
            target = [wrd.upper().translate(str.maketrans(
                '', '', string.punctuation)) for wrd in batch.wrd]
            target_words = [wrd.split(" ") for wrd in target]

            if adv:
                if targeted:
                    self.adv_wer_metric_target.append(
                        ids, predicted_words, target_words
                    )
                    self.adv_cer_metric_target.append(
                        ids, predicted, target
                    )
                    self.adv_ser_metric_target.append(
                        ids, predicted, target)
                else:
                    self.adv_wer_metric.append(
                        ids, predicted_words, target_words)
                    self.adv_cer_metric.append(
                        ids, predicted_words, target_words)
                print('adv_cer =', self.adv_cer_metric.summarize())
            else:
                self.wer_metric.append(ids, predicted_words, target_words)
                self.cer_metric.append(ids, predicted_words, target_words)
                print('cer =', self.cer_metric.summarize())
        if reduction == 'mean':
            loss = loss.mean()
        return loss

    def init_optimizers(self):
        "Initializes the optimizer and model optimizer"
        self.optimizer = self.hparams.opt_class(
            self.hparams.model.parameters()
        )

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("optimizer", self.optimizer)
    
    def on_stage_end(self, stage, stage_loss, epoch, stage_adv_loss=None, stage_adv_loss_target=None):
        super().on_stage_end(stage, stage_loss, epoch, stage_adv_loss, stage_adv_loss_target)
        # Save the WER and CER to file at the end of the test stage.
        if stage == sb.Stage.TEST:
            with open(self.hparams.wer_file.replace("wer", "cer"), "w") as cer:
                self.cer_metric.write_stats(cer)
            with open(f'{self.hparams.wer_file.replace("wer", "cer_adv")}', "w") as cer:
                self.adv_cer_metric.write_stats(cer)
            with open(f'{self.hparams.wer_file.replace("wer", "wer_adv")}', "w") as wer:
                self.adv_wer_metric.write_stats(wer)
