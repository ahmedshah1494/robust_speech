"""
A variation of the Unviersal attack (https://arxiv.org/pdf/1905.03828.pdf), mixed with Projected Gradient Descent
"""

import pdb
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm

from speechbrain.utils.edit_distance import wer_details_for_batch
from robust_speech.models.seq2seq import S2SASR
from robust_speech.models.ctc import CTCASR
from robust_speech.models.modules.deepspeech import DeepspeechCTCASR
from robust_speech.models.sb_hf_binding import HuggingFaceASR

import robust_speech as rs
from robust_speech.adversarial.attacks.attacker import TrainableAttacker
from robust_speech.adversarial.utils import (
    l2_clamp_or_normalize,
    linf_clamp,
    rand_assign,
)

CER_SUCCESS_THRESHOLD = 70
CER_SUCCESS_THRESHOLD_TARGETED = 15
MAXLEN = 288000
MAXLEN_TIME = 20000


def reverse_bound_from_rel_bound(batch, rel, order=np.inf):
    """From a relative eps bound, reconstruct the absolute bound for the given batch"""
    wavs, wav_lens = batch.sig
    wav_lens = [int(wavs.size(1) * r) for r in wav_lens]
    epss = []
    for i in range(len(wavs)):
        eps = torch.norm(wavs[i, : wav_lens[i]], p=order) / rel
        epss.append(eps)
    return torch.tensor(epss).to(wavs.device)


class UniversalAttack(TrainableAttacker):
    """
    Implementation of the Universal attack (https://arxiv.org/pdf/1905.03828.pdf)
    The attack performs nb_iter steps of size eps_iter, while always staying
    within eps from the initial point.

    Arguments
    ---------
    asr_brain: rs.adversarial.brain.ASRBrain
       brain object.
    snr: float
       maximum distortion.
    eps: float
       maximum distortion.
    nb_iter: int
       number of iterations.
    eps_iter: float
       attack step size.
    rand_init: (optional bool)
       random initialization.
    clip_min: (optional) float
       mininum value per input dimension.
    clip_max: (optional) float
       maximum value per input dimension.
    order: (optional) int
       the order of maximum distortion (inf or 2).
    targeted: bool
       if the attack is targeted.
    train_mode_for_backward: bool
       whether to force training mode in backward passes (necessary for RNN models)
    """

    def __init__(
        self,
        asr_brain,
        # eps=0.3,
        snr = 50,
        eps_item=0.1,
        nb_epochs=10,
        nb_iter=40,
        rand_init=True,
        clip_min=None,
        clip_max=None,
        order=np.inf,
        targeted=False,
        train_mode_for_backward=True,
        lr=0.001,
        time_universal=False,
        univ_perturb=None,
        checkpointer=None,
    ):
        self.clip_min = clip_min
        self.clip_max = clip_max
        # self.eps = eps
        self.eps_item = eps_item
        self.lr = lr
        self.nb_iter = nb_iter
        self.nb_epochs = nb_epochs
        self.rand_init = rand_init
        self.order = order
        self.targeted = targeted
        self.asr_brain = asr_brain
        self.train_mode_for_backward = train_mode_for_backward
        self.time_universal = time_universal
        self.rel_eps = torch.pow(torch.tensor(10.0), float(snr) / 20)
        self.checkpointer = checkpointer

        self.univ_perturb = univ_perturb
        if checkpointer is not None:
            ckps = checkpointer.list_checkpoints()
            self.nb_epochs = max(self.nb_epochs-len(ckps), 0)
            print(f'Found {len(ckps)} checkpoints, reducing epochs to {self.nb_epochs}')
            self.univ_perturb = checkpointer.recoverables['delta']
            ckp = checkpointer.recover_if_possible()
            print(f'Loaded checkpoint {ckp}')
            if ckp is not None:
                checkpointer.load_checkpoint(ckp)
        elif self.univ_perturb is None:
            len_delta = MAXLEN_TIME if time_universal else MAXLEN
            self.univ_perturb = rs.adversarial.utils.TensorModule(
                size=(len_delta,))
        print("||delta|| =", torch.norm(self.univ_perturb.tensor))

    def fit(self, loader):
        return self._compute_universal_perturbation(loader)

    def _compute_universal_perturbation(self, loader):
        if isinstance(self.asr_brain, S2SASR):
            tokenizer = self.asr_brain.tokenizer if self.asr_brain.tokenizer is not None else self.asr_brain.hparams.tokenizer
            decode = tokenizer.decode_ids
        elif isinstance(self.asr_brain, DeepspeechCTCASR):
            decode = lambda x: x[0]
        elif isinstance(self.asr_brain, CTCASR):
            tokenizer = self.asr_brain.tokenizer if self.asr_brain.tokenizer is not None else self.asr_brain.hparams.tokenizer
            decode = tokenizer.decode_ndim
        elif isinstance(self.asr_brain, HuggingFaceASR):
            tokenizer = self.asr_brain.tokenizer if self.asr_brain.tokenizer is not None else self.asr_brain.hparams.tokenizer
            decode = lambda x : tokenizer.decode(x, skip_special_tokens=True)
        else:
            print(f"WARNING: Brain class not in [S2SASR, DeepspeechCTCASR, CTCASR, HuggingFaceASR]. Assuming decoding is performed in the brain class. Setting decode to identity function.")
            decode = lambda x : x

        if self.train_mode_for_backward:
            self.asr_brain.module_train()
        else:
            self.asr_brain.module_eval()

        delta = self.univ_perturb.tensor.data
        success_rate = 0

        best_success_rate = -100
        epoch = 0

        #####HYPERPARAM for fixed delta#####
        use_time_universal = self.time_universal
        ####################################

        print('Estimating average eps over all training samples')
        self.eps = 0.
        for idx, batch in enumerate(tqdm(loader, dynamic_ncols=True)):
            batch = batch.to(self.asr_brain.device)
            batch_eps = reverse_bound_from_rel_bound(batch, self.rel_eps).min()
            self.eps = (self.eps*idx + batch_eps)/(idx+1)
        print(f'EPS={self.eps}')

        while epoch < self.nb_epochs:
            print(f'{epoch}s epoch')
            epoch += 1
            # GENERATE CANDIDATE FOR UNIVERSAL PERTURBATION
            for idx, batch in enumerate(tqdm(loader, dynamic_ncols=True)):
                batch = batch.to(self.asr_brain.device)
                wav_init, wav_lens = batch.sig

                if use_time_universal:
                    base_delta = delta[:MAXLEN_TIME]
                    delta_x = base_delta.repeat(torch.ceil(
                        wav_init.shape[1]/base_delta.shape[0]))
                    delta_x = delta_x[:wav_init.shape[1]]
                else:
                    # Slice or Pad to match the shape with data point x
                    delta_x = torch.zeros_like(wav_init[0])
                    if wav_init.shape[1] <= delta.shape[0]:
                        delta_x[:wav_init.shape[1]
                                ] = delta[: wav_init.shape[1]].detach()
                    else:
                        delta_x[: delta.shape[0]] = delta.detach()
                delta_batch = delta_x.unsqueeze(0).expand(wav_init.size())
                _, predicted_tokens_origin, _ = self.asr_brain.compute_forward(
                    batch, rs.Stage.ADVTRUTH)
                predicted_words_origin = [
                    decode(utt_seq).split(" ")
                    for utt_seq in predicted_tokens_origin
                ]

                if use_time_universal:
                    r = torch.rand_like(base_delta) / 1e+4
                else:
                    r = torch.rand_like(delta_x) / 1e+4
                r.requires_grad_()

                batch.sig = wav_init + delta_batch, wav_lens
                _, predicted_tokens_adv, _ = self.asr_brain.compute_forward(
                    batch, rs.Stage.ADVTARGET)
                predicted_words_adv = [
                    decode(utt_seq).split(" ")
                    for utt_seq in predicted_tokens_adv
                ]

                # self.asr_brain.cer_metric.append(batch.id, predicted_words_adv, predicted_words_origin)
                def cer_metric(id, ref, hyp):
                    hyp = [['UNK'] if ((len(h) == 0) or ((len(h) == 1) and h[0]=='')) else h for h in hyp]
                    computer = self.asr_brain.hparams.cer_computer()
                    try:
                        computer.append(id, ref, hyp)
                        return computer.summarize("error_rate")
                    except Exception as e:
                        print(e)
                        print(ref)
                        print(hyp)
                        return 100
                CER = 0
                # print(CER)

                for i in range(self.nb_iter):
                    if use_time_universal:
                        r_batch = r.repeat(torch.ceil(wav_init.shape[1]/r.shape[0]))[
                            :delta_x.size()].unsqueeze(0).expand(delta_batch.size())
                    else:
                        r_batch = r.unsqueeze(0).expand(delta_batch.size())

                    batch.sig = wav_init + delta_batch + r_batch, wav_lens
                    predictions = self.asr_brain.compute_forward(
                        batch, rs.Stage.ATTACK)
                    # loss = 0.5 * r.norm(dim=1, p=2) - self.asr_brain.compute_objectives(predictions, batch, rs.Stage.ATTACK)
                    ctc = - \
                        self.asr_brain.compute_objectives(
                            predictions, batch, rs.Stage.ATTACK)
                    l2_norm = r.norm(p=2).to(
                        self.asr_brain.device)
                    loss = 0.5 * l2_norm + ctc
                    # loss = ctc
                    loss.backward(inputs=r)
                    # print(l2_norm,ctc,CER)
                    grad_sign = r.grad.data.sign()
                    r.data = r.data - self.lr * grad_sign
                    # r.data = r.data - 0.1 * r.grad.data
                    r.data = linf_clamp(r.data, self.eps_item)
                    r.data = linf_clamp(
                        delta_x + r.data, self.eps) - delta_x

                    # print("delta's mean : ", torch.mean(delta_x).data)
                    # print("r's mean : ",torch.mean(r).data)
                    r.grad.data.zero_()

                    _, predicted_tokens_adv, _ = self.asr_brain.compute_forward(
                        batch, rs.Stage.ADVTARGET)
                    predicted_words_adv = [
                        decode(utt_seq).split(" ")
                        for utt_seq in predicted_tokens_adv
                    ]
                    
                    CER = cer_metric(batch.id, predicted_words_origin,
                                     predicted_words_adv)
                    # print(CER)
                    if CER >= CER_SUCCESS_THRESHOLD:
                        break

                # print(f'CER = {CER}')
                delta_x = linf_clamp(delta_x + r.data, self.eps)

                if delta.shape[0] <= delta_x.shape[0]:
                    delta = delta_x[:delta.shape[0]].detach()
                else:
                    delta[:delta_x.shape[0]] = delta_x.detach()

            # print(f'MAX OF INPUT WAVE IS {torch.max(wav_init).data}')
            # print(f'AVG OF INPUT WAVE IS {torch.mean(wav_init).data}')
            # print(f'MAX OF DELTA IS {torch.max(delta).data}')
            # print(f'AVG OF DELTA IS {torch.mean(delta).data}')
            print('CHECK SUCCESS RATE OVER ALL TRAINING SAMPLES')
            # TO CHECK SUCCESS RATE OVER ALL TRAINING SAMPLES
            total_sample = 0.
            fooled_sample = 0.

            cer_computer = self.asr_brain.hparams.cer_computer()

            for idx, batch in enumerate(tqdm(loader, dynamic_ncols=True)):
                batch = batch.to(self.asr_brain.device)
                wav_init, wav_lens = batch.sig

                if use_time_universal:
                    base_delta = delta[:MAXLEN_TIME]
                    delta_x = base_delta.repeat(torch.ceil(
                        wav_init.shape[1]/base_delta.shape[0]))
                    delta_x = delta_x[:wav_init.shape[1]]
                else:
                    delta_x = torch.zeros_like(wav_init[0])
                    if wav_init.shape[1] <= delta.shape[0]:
                        delta_x = delta[:wav_init.shape[1]]
                    else:
                        delta_x[:delta.shape[0]] = delta
                # if idx == 400:
                #     break
                #     raise NotImplementedError

                # CER(Xi)
                delta_batch = delta_x.unsqueeze(0).expand(wav_init.size())
                _, predicted_tokens_origin, _ = self.asr_brain.compute_forward(
                    batch, rs.Stage.ADVTRUTH)

                predicted_words_origin = [
                    decode(utt_seq).split(" ")
                    for utt_seq in predicted_tokens_origin
                ]

                # CER(Xi + v)
                batch.sig = wav_init + delta_batch.to(self.asr_brain.device), wav_lens
                _, predicted_tokens_adv, _ = self.asr_brain.compute_forward(
                    batch, rs.Stage.ADVTRUTH)
                predicted_words_adv = [
                    decode(utt_seq).split(" ")
                    for utt_seq in predicted_tokens_adv
                ]
                predicted_words_adv = [['UNK'] if ((len(h) == 0) or ((len(h) == 1) and h[0]=='')) else h for h in predicted_words_adv]
                try:
                    cer_computer.append(batch.id, predicted_words_origin,
                                    predicted_words_adv)
                except Exception as e:
                    print(e)
                    print(predicted_words_origin)
                    print(predicted_words_adv)
                total_sample += 1.
            success_rate = cer_computer.summarize("error_rate")
            print(f'SUCCESS RATE (CER) IS {success_rate:.4f}')
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                if use_time_universal:
                    self.univ_perturb.tensor.data = base_delta.detach()
                else:
                    self.univ_perturb.tensor.data = delta.detach()
                print(
                    f"Perturbation vector with best success rate saved. Success rate:{best_success_rate:.2f}%")
                self.checkpointer.save_checkpoint()
        print(
            f"Training finisihed. Best success rate: {best_success_rate:.2f}%")

    def perturb(self, batch):
        """
        Compute an adversarial perturbation
        Arguments
        ---------
        batch : sb.PaddedBatch
           The input batch to perturb

        Returns
        -------
        the tensor of the perturbed batch
        """
        if self.train_mode_for_backward:
            self.asr_brain.module_train()
        else:
            self.asr_brain.module_eval()

        save_device = batch.sig[0].device
        batch = batch.to(self.asr_brain.device)
        save_input = batch.sig[0]
        wav_init = torch.clone(save_input)

        delta = self.univ_perturb.tensor.data.to(self.asr_brain.device)

        if self.time_universal:
            delta_x = torch.zeros_like(wav_init[0])
            temp_delta = torch.concat(
                (delta, delta), axis=0)
            # Concat delta to match length with data point x
            while wav_init.shape[1] > temp_delta.shape[1]:
                temp_delta = torch.concat(
                    (temp_delta, temp_delta), axis=1)
            delta_x[:wav_init.shape[1]] = temp_delta[:, :wav_init.shape[1]]
        else:
            if wav_init.shape[1] <= delta.shape[0]:
                delta_x = delta[:wav_init.shape[1]]
            else:
                delta_x = torch.zeros_like(wav_init[0])
                delta[:delta.shape[0]] = delta
        delta_batch = delta_x.unsqueeze(0).expand(wav_init.size())
        wav_adv = wav_init + delta_batch
        # self.eps = 1.0
        batch.sig = save_input, batch.sig[1]
        batch = batch.to(save_device)
        self.asr_brain.module_eval()
        return wav_adv.data.to(save_device)
