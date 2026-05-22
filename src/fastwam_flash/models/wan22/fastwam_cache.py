from typing import Any, Optional

import torch
import numpy as np

from fastwam_flash.utils.logging_config import get_logger

from .fastwam import FastWAM

from .action_dit import ActionDiT
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


infer_action_mapping = {
    'none': 'infer_action',
    'naivecache': 'infer_action_with_naivecache',
    'teacache': 'infer_action_with_teacache',
    'dreamzero': 'infer_action_with_dreamzero',
    'blockcache': 'infer_action_with_blockcache',
    'speccache': 'infer_action_with_speccache',
    'branchcache': 'infer_action_with_branchcache',
    'parablock': 'infer_action_with_parablock',
}

class FastWAMCache(FastWAM):
    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        dit_cache_config: dict[str, Any] | None = None,
    ):
        super().__init__(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_dim=text_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
        )
        
        # for cache
        cache_type = dit_cache_config.get("cache_type", "none")
        assert cache_type in [
            "none",
            "naivecache",
            "teacache",
            "dreamzero",
            "blockcache",
            "speccache",
            "branchcache",
            "parablock",
        ], f"Error: unsupported cache type {cache_type}"
        self.cache_type = cache_type
        self.infer_action = getattr(self, infer_action_mapping[cache_type])
        self.register_cache_config(dit_cache_config)
    
    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        dit_cache_config: dict[str, Any] | None = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            dit_cache_config=dit_cache_config,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model
    
    def register_cache_config(self, dit_cache_config):
        # for naivecache
        if self.cache_type == "naivecache":
            assert "naivecache_config" in dit_cache_config, "naivecache_config is required for naivecache"
            self.naivecache_config = dit_cache_config["naivecache_config"]
        
        # for teachache
        if self.cache_type == "teacache":
            assert "teacache_config" in dit_cache_config, "teacache_config is required for teacache"
            self.teacache_config = dit_cache_config["teacache_config"]
            self.rescale_func = np.poly1d(self.teacache_config["coefficients"])
        
        # for dreamzero
        if self.cache_type == "dreamzero":
            assert "dreamzero_config" in dit_cache_config, "dreamzero_config is required for dreamzero cache"
            self.dreamzero_config = dit_cache_config["dreamzero_config"]
        
        # for blockcache
        if self.cache_type == "blockcache":
            assert "blockcache_config" in dit_cache_config, "blockcache_config is required for blockcache cache"
            self.blockcache_config = dit_cache_config["blockcache_config"]
        
        # for speccache
        if self.cache_type == "speccache":
            assert "speccache_config" in dit_cache_config, "speccache_config is required for speccache cache"
            self.speccache_config = dit_cache_config["speccache_config"]
            assert 0 in self.speccache_config['batch1_cal_steps'] and 0 in self.speccache_config['batch2_cal_steps'], "the first step must be calculated in both batches"
            if len(self.speccache_config['batch1_cal_steps']) > len(self.speccache_config['batch2_cal_steps']):
                self.speccache_config['batch1_cal_steps'], self.speccache_config['batch2_cal_steps'] = self.speccache_config['batch2_cal_steps'], self.speccache_config['batch1_cal_steps']
            self.reference_action = None
            self.replan_steps = 10
        
        # for branchcache
        if self.cache_type == "branchcache":
            assert "branchcache_config" in dit_cache_config, "branchcache_config is required for branchcache cache"
            self.branchcache_config = dit_cache_config["branchcache_config"]
            self.branch_steps_list = self.branchcache_config['branch_steps_list']
            self.branch_merge_policy = self.branchcache_config['branch_merge_policy'] if "branch_merge_policy" in self.branchcache_config else "last"
        
        # for parablock
        if self.cache_type == "parablock":
            assert "parablock_config" in dit_cache_config, "parablock_config is required for parablock cache"
            self.parablock_config = dit_cache_config["parablock_config"]
            self.recompute_threshold = self.parablock_config['recompute_threshold']
    
    def reset_episode(self):
        if hasattr(self, "reference_action"):
            self.reference_action = None
    
    @torch.no_grad()
    def infer_action_with_naivecache(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )

        step_idx = 0
        step_without_cache = 0
        step_with_cache = 0
        prev_pred = None

        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            step_without_cache += 1

            if step_idx in self.naivecache_config["cal_steps"] or prev_pred is None:
                step_with_cache += 1
                timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

                if not self.naivecache_config["add_blockcurvecache"]:
                    pred_action_posi = self._predict_action_noise_with_cache(
                        latents_action=latents_action,
                        timestep_action=timestep_action,
                        context=context,
                        context_mask=context_mask,
                        video_kv_cache=video_kv_cache,
                        attention_mask=attention_mask,
                        video_seq_len=video_seq_len,
                    )
                else:
                    pred_action_posi = self._predict_action_noise_with_cache_and_blockcurvecache(
                        latents_action=latents_action,
                        timestep_action=timestep_action,
                        context=context,
                        context_mask=context_mask,
                        video_kv_cache=video_kv_cache,
                        attention_mask=attention_mask,
                        video_seq_len=video_seq_len,
                        enable_blockcache=step_idx != 0 and step_idx != num_inference_steps -1,
                    )
                pred_action = pred_action_posi
                prev_pred = pred_action.clone().detach()
            else:
                pred_action = prev_pred

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            step_idx += 1
        
        logger.info(f"runned {step_with_cache}/{step_without_cache} steps")

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
    
    # for TeaCache
    @torch.no_grad()
    def infer_action_with_teacache(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )

        prev_residual, prev_time_modulate, accumulated_distance = None, None, 0.0
        step_idx = 0
        step_without_cache = 0
        step_with_cache = 0

        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            step_without_cache += 1
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            action_pre = self.action_expert.pre_dit(
                action_tokens=latents_action,
                timestep=timestep_action,
                context=context,
                context_mask=context_mask,
            )

            time_modulate = action_pre["t_mod"].float().clone().detach()

            should_cal = False
            if step_idx < self.teacache_config["step_start"] or step_idx >= self.teacache_config["step_end"] or prev_residual is None:
                should_cal = True
            else:
                accumulated_distance += self.rescale_func(
                    ((time_modulate - prev_time_modulate).abs().mean() / prev_time_modulate.abs().mean()).cpu().item()
                )
                if accumulated_distance > self.teacache_config['threshold']:
                    should_cal = True

            pred_action_posi, pred_residual = self._predict_action_noise_with_cache_and_teacache(
                action_pre=action_pre,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                prev_residual=prev_residual,
                should_cal=should_cal,
            )
            pred_action = pred_action_posi

            if should_cal:
                prev_time_modulate = time_modulate
                prev_residual = pred_residual
                accumulated_distance = 0.0
                step_with_cache += 1

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            step_idx += 1

        logger.info(f"runned {step_with_cache}/{step_without_cache} steps")
        
        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def _predict_action_noise_with_cache_and_teacache(
        self,
        action_pre,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        prev_residual: torch.Tensor | None,
        should_cal: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_tokens, pred_residual = self.mot.forward_action_with_video_cache_and_teacache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            prev_residual=prev_residual,
            should_cal=should_cal,
        )
        return self.action_expert.post_dit(action_tokens, action_pre), pred_residual

    # for cache from DreamZero(based on cos similarity)
    @torch.no_grad()
    def infer_action_with_dreamzero(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )

        # Initialize variables for DiT cache
        prev_predictions = []
        skip_countdown = 0
        step_without_cache = 0
        step_with_cache = 0
        cal_steps = []
        
        for i, (step_t_action, step_delta_action) in enumerate(zip(infer_timesteps_action, infer_deltas_action)):
            step_without_cache += 1

            # Check if we should skip this step
            if skip_countdown > 0:
                skip_countdown -= 1
            
            if skip_countdown == 0:
                step_with_cache += 1
                cal_steps.append(i)
                timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

                pred_action_posi = self._predict_action_noise_with_cache(
                    latents_action=latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                )
                pred_action = pred_action_posi

                # Store the current prediction
                prev_predictions.append((timestep_action, pred_action))
                
                # Calculate similarity with previous prediction if we have enough history
                if len(prev_predictions) >= 2:
                    v_last = prev_predictions[-1][1].flatten(1).float()
                    v_prev = prev_predictions[-2][1].flatten(1).float()
                    sim = torch.nn.functional.cosine_similarity(v_last, v_prev, dim=1).mean()
                    
                    # Check if similarity is high enough to skip steps
                    thresholds = self.dreamzero_config["thresholds"]  # [0.95, 0.93]
                    countdowns = self.dreamzero_config["countdowns"]  # [4, 2]
                    
                    for threshold, countdown in zip(thresholds, countdowns):
                        if sim > threshold:
                            skip_countdown = countdown
                            break
            
            else:
                pred_action = prev_predictions[-1][1]

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
        
        logger.info(f"runned {step_with_cache}/{step_without_cache} steps ({cal_steps})")

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
    
    # for BlockCache
    @torch.no_grad()
    def infer_action_with_blockcache(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )

        step_idx = 0

        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache_and_blockcache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                enable_blockcache=step_idx % self.blockcache_config["interval"] != 0,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            step_idx += 1

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
    
    @torch.no_grad()
    def _predict_action_noise_with_cache_and_blockcache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        enable_blockcache: bool = False,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache_and_blockcache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            enable_blockcache=enable_blockcache,
            blockcache_ratio=self.blockcache_config["ratio"],
        )
        return self.action_expert.post_dit(action_tokens, action_pre)
    
    @torch.no_grad()
    def _predict_action_noise_with_cache_and_blockcurvecache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        enable_blockcache: bool = False,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache_and_blockcurvecache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            enable_blockcache=enable_blockcache
        )
        return self.action_expert.post_dit(action_tokens, action_pre)
    
    # for SpecCache
    @torch.no_grad()
    def infer_action_with_speccache(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        accept_signal: Optional[bool] = None,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )

        total_steps = len(infer_timesteps_action)  # len(zip(infer_timesteps_action, infer_deltas_action))
        batch1_step_idx, batch2_step_idx = 0, 0
        batch1_prev_pred, batch2_prev_pred = None, None
        batch1_latents_action, batch2_latents_action = latents_action, latents_action.clone()
        batch1_should_cal, batch2_should_cal = False, False

        if context is not None:
            batch_context = context.expand(2, *context.shape[1:])
        if context_mask is not None:
            batch_context_mask = context_mask.expand(2, *context_mask.shape[1:])

        while batch1_step_idx < total_steps:  # or batch2_step_idx < total_steps:
            batch1_should_cal = batch1_step_idx in self.speccache_config['batch1_cal_steps']
            batch2_should_cal = batch2_step_idx in self.speccache_config['batch2_cal_steps']
            should_cal = batch1_should_cal and batch2_should_cal

            if should_cal:
                batch1_step_t_action, batch1_step_delta_action = infer_timesteps_action[batch1_step_idx], infer_deltas_action[batch1_step_idx]
                batch2_step_t_action, batch2_step_delta_action = infer_timesteps_action[batch2_step_idx], infer_deltas_action[batch2_step_idx]
                batch1_timestep_action = batch1_step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
                batch2_timestep_action = batch2_step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
                combined_latents_action = torch.cat([batch1_latents_action, batch2_latents_action], dim=0)
                combined_timestep_action = torch.cat([batch1_timestep_action, batch2_timestep_action], dim=0)
                pred_action_posi = self._predict_action_noise_with_cache(
                    latents_action=combined_latents_action,
                    timestep_action=combined_timestep_action,
                    context=batch_context,
                    context_mask=batch_context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                )
                batch1_pred_action, batch2_pred_action = pred_action_posi.chunk(2, dim=0)
                batch1_prev_pred = batch1_pred_action.clone()
                batch2_prev_pred = batch2_pred_action.clone()
                batch1_latents_action = self.infer_action_scheduler.step(batch1_pred_action, batch1_step_delta_action, batch1_latents_action)
                batch1_step_idx += 1
                batch2_latents_action = self.infer_action_scheduler.step(batch2_pred_action, batch2_step_delta_action, batch2_latents_action)
                batch2_step_idx += 1
                continue

            if not batch1_should_cal:
                batch1_latents_action = self.infer_action_scheduler.step(batch1_prev_pred, infer_deltas_action[batch1_step_idx], batch1_latents_action)
                batch1_step_idx += 1
            if not batch2_should_cal:
                batch2_latents_action = self.infer_action_scheduler.step(batch2_prev_pred, infer_deltas_action[batch2_step_idx], batch2_latents_action)
                batch2_step_idx += 1
        
        while batch2_step_idx < total_steps:
            step_t_action, step_delta_action = infer_timesteps_action[batch2_step_idx], infer_deltas_action[batch2_step_idx]
            if batch2_step_idx in self.speccache_config['batch2_cal_steps']:
                timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)
                pred_action_posi = self._predict_action_noise_with_cache(
                    latents_action=batch2_latents_action,
                    timestep_action=timestep_action,
                    context=context,
                    context_mask=context_mask,
                    video_kv_cache=video_kv_cache,
                    attention_mask=attention_mask,
                    video_seq_len=video_seq_len,
                )
                pred_action = pred_action_posi
                batch2_prev_pred = pred_action.clone()
                batch2_latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, batch2_latents_action)
            else:
                batch2_latents_action = self.infer_action_scheduler.step(batch2_prev_pred, step_delta_action, batch2_latents_action)
            batch2_step_idx += 1
        
        draft_action, action = batch1_latents_action[0], batch2_latents_action[0]
        
        if accept_signal is not None:
            accept_draft = accept_signal
        else:
            assert action.ndim == 2 and action.shape[1] == 7, f"latents_action.shape: {action.shape}, should be (X, 7)"
            reference_steps = min(self.replan_steps, action.shape[0] - self.replan_steps)
            accept_draft = False
            if self.reference_action is None:
                accept_draft = True
            else:
                accept_draft = self.is_draft_accepted(draft_action[:reference_steps], self.reference_action[:reference_steps])
            self.reference_action = action[self.replan_steps:].clone()

        logger.info(f"accept draft? {accept_draft}")

        return {
            # "draft_action": draft_action.detach().to(device="cpu", dtype=torch.float32),
            "draft_action": draft_action.detach().to(device="cpu", dtype=torch.float32),
            "orig_action": action.detach().to(device="cpu", dtype=torch.float32),
            "action": draft_action.detach().to(device="cpu", dtype=torch.float32) if accept_draft else action.detach().to(device="cpu", dtype=torch.float32),
        }
    
    def is_draft_accepted(self, draft_action, reference_action):
        assert draft_action.shape == reference_action.shape and draft_action.shape[1] == 7, f"draft_action.shape: {draft_action.shape}, reference_action.shape: {reference_action.shape}"
        delta_action = (draft_action - reference_action).mean(dim=0).abs()
        xyz_delta, rot_delta = delta_action[:3], delta_action[3:6]
        return xyz_delta.max() < 0.05 and rot_delta.max() < 0.03
    
    # for branchcache (initial version for testing)
    @torch.no_grad()
    def infer_action_with_branchcache(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )

        if self.branch_merge_policy == 'batchstep':
            # check branch_steps_list:
            for branch_steps in self.branch_steps_list:
                if not isinstance(branch_steps, list):
                    raise ValueError(f"Each element in branch_steps_list must be a list, got {type(branch_steps)}")
                if not branch_steps or branch_steps[0] != 0 or len(branch_steps) != 2:
                    raise ValueError(f"Each branch_steps must be [0, x], got {branch_steps}")
                x = branch_steps[-1]
                if not isinstance(x, int) or not (1 <= x <= num_inference_steps - 1):
                    raise ValueError(f"The last element x in branch_steps must be an integer between 1 and {num_inference_steps - 1}, got {x}")
            # Sort by x (last element) in ascending order
            self.branch_steps_list.sort(key=lambda bs: bs[-1])

            num_batch = len(self.branch_steps_list)
            if context is not None:
                batch_context = context.expand(num_batch, *context.shape[1:])
            if context_mask is not None:
                batch_context_mask = context_mask.expand(num_batch, *context_mask.shape[1:])
            batch_step_idx = [0 for _ in range(num_batch)]
            batch_latents_action = [latents_action.clone() for _ in range(num_batch)]
            batch_prev_pred = [None for _ in range(num_batch)]

            # stage 1 (puiblic step 0):
            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=torch.cat(batch_latents_action, dim=0),
                timestep_action=torch.cat(
                    [infer_timesteps_action[i].unsqueeze(0).to(dtype=latents_action.dtype, device=self.device) for i in batch_step_idx],
                    dim=0
                ),
                context=batch_context,
                context_mask=batch_context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            for i in range(num_batch):
                batch_pred_action_i = pred_action_posi.chunk(num_batch, dim=0)[i]
                batch_prev_pred[i] = batch_pred_action_i.clone()
                batch_latents_action[i] = self.infer_action_scheduler.step(
                    batch_pred_action_i,
                    infer_deltas_action[batch_step_idx[i]],
                    batch_latents_action[i]
                )
                batch_step_idx[i] += 1

            # stage 2 (sync before next step):
            for i in range(num_batch):
                while batch_step_idx[i] < self.branch_steps_list[i][1]:
                    batch_latents_action[i] = self.infer_action_scheduler.step(
                        batch_prev_pred[i],
                        infer_deltas_action[batch_step_idx[i]],
                        batch_latents_action[i]
                    )
                    batch_step_idx[i] += 1
            
            # stage 3 (next public step):
            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=torch.cat(batch_latents_action, dim=0),
                timestep_action=torch.cat(
                    [infer_timesteps_action[i].unsqueeze(0).to(dtype=latents_action.dtype, device=self.device) for i in batch_step_idx],
                    dim=0
                ),
                context=batch_context,
                context_mask=batch_context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            batch_final_pred = {i: pred_action_posi.chunk(num_batch, dim=0)[i] for i in range(num_batch)}

            # stage 4 (finish remaining steps of the first branch):
            current_pred = batch_prev_pred[0]
            while batch_step_idx[0] < num_inference_steps:
                if batch_step_idx[0] in batch_final_pred.keys():
                    current_pred = batch_final_pred[batch_step_idx[0]]
                batch_latents_action[0] = self.infer_action_scheduler.step(
                    current_pred,
                    infer_deltas_action[batch_step_idx[0]],
                    batch_latents_action[0]
                )
                batch_step_idx[0] += 1

        else:
            branch_latents_actions = []
            final_branch_latents_actions = []
            final_latents_action = latents_action.clone().detach()
            for branch_steps in self.branch_steps_list:
                current_latents_action = latents_action.clone().detach()
                latents_action_records = []
                prev_pred = None

                num_step = 0
                for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
                    if num_step in branch_steps or prev_pred is None:
                        timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

                        pred_action_posi = self._predict_action_noise_with_cache(
                            latents_action=current_latents_action,
                            timestep_action=timestep_action,
                            context=context,
                            context_mask=context_mask,
                            video_kv_cache=video_kv_cache,
                            attention_mask=attention_mask,
                            video_seq_len=video_seq_len,
                        )
                        pred_action = pred_action_posi
                        prev_pred = pred_action.clone().detach()
                    else:
                        pred_action = prev_pred
                    
                    current_latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, current_latents_action)
                    latents_action_records.append(current_latents_action[0].clone().detach().to(device="cpu", dtype=torch.float32))
                    num_step += 1
                final_latents_action = current_latents_action
                final_branch_latents_actions.append(final_latents_action)
                branch_latents_actions.append(latents_action_records)
        
        if self.branch_merge_policy == 'last':
            output_action = final_latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        elif self.branch_merge_policy == 'mean':
            output_action = torch.stack(final_branch_latents_actions, dim=0).mean(dim=0)[0].detach().to(device="cpu", dtype=torch.float32)
        elif self.branch_merge_policy == 'batchstep':
            output_action = batch_latents_action[0][0].detach().to(device="cpu", dtype=torch.float32)
            branch_latents_actions = None
        else:
            raise ValueError(f"Unknown branch_merge_policy: {self.branch_merge_policy}")

        return {
            "action": output_action,
            "branch_latents_action": branch_latents_actions,
        }
    
    # for ParaBlock
    @torch.no_grad()
    def infer_action_with_parablock(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )

        step_idx = 0

        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache_and_parablock(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
                recompute_threshold=self.recompute_threshold,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            step_idx += 1

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }
    
    @torch.no_grad()
    def _predict_action_noise_with_cache_and_parablock(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
        recompute_threshold: float = 0.25,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens, cal_layer_num = self.mot.forward_action_with_video_cache_and_parablock(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
            recompute_threshold=recompute_threshold,
        )

        logger.info(f"cal_layer_num: {cal_layer_num}")

        return self.action_expert.post_dit(action_tokens, action_pre)