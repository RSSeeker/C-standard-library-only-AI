# AI_C/c —— Python 版神经网络工具箱的 C 移植
#
# 如果你的环境没有 make，直接编译即可（Windows/MinGW 与 Linux/macOS 通用）：
#   gcc -std=c17 -O2 -Wall -Wextra -o demo demo.c ai.c tensor.c layers_conv.c \
#       layers_norm.c rnn.c attention.c llm.c containers.c json.c model_io.c text.c optim.c \
#       sched.c -lm
#
# 常用目标：
#   make            构建全部程序
#   make test       构建并运行全部 C 自测
#   make check      与 Python 版做数值等价比对（需要 python 与环境里的 AI.py）
#   make clean

CC      ?= gcc
CFLAGS  ?= -std=c17 -O2 -Wall -Wextra
LDLIBS  ?= -lm

# 核心库 = 除各程序入口之外的所有 .c（新增模块后无需改这里）
MAINS   = demo.c bench.c compare_python.c check_noalloc.c
CORE    = $(filter-out $(MAINS), $(wildcard *.c))

BINS    = demo bench compare_python check_noalloc
TESTS   = $(patsubst tests/%.c,%,$(wildcard tests/test_*.c))

all: $(BINS) $(addprefix tests/,$(TESTS))

%: %.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) $(LDLIBS)

tests/%: tests/%.c $(CORE)
	$(CC) $(CFLAGS) -o $@ $< $(CORE) $(LDLIBS)

# 训练循环零分配检查需要链接器的 --wrap 支持（GNU ld / lld）
check_noalloc: check_noalloc.c $(CORE)
	$(CC) $(CFLAGS) -Wl,--wrap=malloc -Wl,--wrap=calloc -Wl,--wrap=realloc \
		-o $@ $< $(CORE) $(LDLIBS)

test: $(addprefix tests/,$(TESTS))
	@echo "=== C 自测 ==="
	@for t in $(TESTS); do \
		echo "--- $$t ---"; \
		./tests/$$t self || exit 1; \
	done
	@echo "全部通过"

check: compare_python
	@echo "=== 与 Python 版数值等价比对 ==="
	python tools/c_reference.py reference.txt
	./compare_python reference.txt
	python tools/json_interop.py write
	$(MAKE) tests/test_json && ./tests/test_json write
	python tools/json_interop.py check
	./tests/test_json check
	python tools/text_interop.py
	python tools/sched_interop.py
	python tools/norm_reference.py norm_reference.json
	$(MAKE) tests/compare_norm && ./tests/compare_norm norm_reference.json
	python tools/adamw_reference.py adamw_reference.json
	$(MAKE) tests/compare_adamw && ./tests/compare_adamw adamw_reference.json
	python tools/lion_reference.py lion_reference.json
	$(MAKE) tests/compare_lion && ./tests/compare_lion lion_reference.json
	python tools/adafactor_reference.py adafactor_reference.json
	$(MAKE) tests/compare_adafactor && ./tests/compare_adafactor adafactor_reference.json
	python tools/sched_extra_reference.py sched_extra_reference.json
	$(MAKE) tests/compare_sched_extra && ./tests/compare_sched_extra sched_extra_reference.json
	python tools/rmsnorm_reference.py rmsnorm_reference.json
	$(MAKE) tests/compare_rmsnorm && ./tests/compare_rmsnorm rmsnorm_reference.json
	python tools/activations_reference.py activations_reference.json
	$(MAKE) tests/compare_activations && ./tests/compare_activations activations_reference.json
	python tools/loss_reference.py loss_reference.json
	$(MAKE) tests/compare_loss && ./tests/compare_loss loss_reference.json
	python tools/metrics_reference.py metrics_reference.json
	$(MAKE) tests/compare_metrics && ./tests/compare_metrics metrics_reference.json
	python tools/text_metrics_reference.py text_metrics_reference.json
	$(MAKE) tests/compare_text_metrics && ./tests/compare_text_metrics text_metrics_reference.json
	python tools/audio_metrics_reference.py audio_metrics_reference.json
	$(MAKE) tests/compare_audio_metrics && ./tests/compare_audio_metrics audio_metrics_reference.json
	python tools/visual_metrics_reference.py visual_metrics_reference.json
	$(MAKE) tests/compare_visual_metrics && ./tests/compare_visual_metrics visual_metrics_reference.json
	python tools/ctc_beam_reference.py ctc_beam_reference.json
	$(MAKE) tests/compare_ctc_beam && ./tests/compare_ctc_beam ctc_beam_reference.json
	python tools/onnx_mlp_reference.py onnx_mlp_reference.json
	$(MAKE) tests/compare_onnx_mlp && ./tests/compare_onnx_mlp onnx_mlp_reference.json
	python tools/offline_rl_reference.py offline_rl_reference.json
	$(MAKE) tests/compare_offline_rl && ./tests/compare_offline_rl offline_rl_reference.json
	python tools/egnn_reference.py egnn_reference.json
	$(MAKE) tests/compare_egnn && ./tests/compare_egnn egnn_reference.json
	python tools/spec_decode_reference.py spec_decode_reference.json
	$(MAKE) tests/compare_spec_decode && ./tests/compare_spec_decode spec_decode_reference.json
	python tools/diffusion_reference.py diffusion_reference.json
	$(MAKE) tests/compare_diffusion && ./tests/compare_diffusion diffusion_reference.json
	python tools/flowmatching_reference.py flowmatching_reference.json
	$(MAKE) tests/compare_flowmatching && ./tests/compare_flowmatching flowmatching_reference.json
	python tools/tts_reference.py tts_reference.json
	$(MAKE) tests/compare_tts && ./tests/compare_tts tts_reference.json
	python tools/pix2pix_reference.py pix2pix_reference.json
	$(MAKE) tests/compare_pix2pix && ./tests/compare_pix2pix pix2pix_reference.json
	python tools/gan_variants_reference.py gan_variants_reference.json
	$(MAKE) tests/compare_gan_variants && ./tests/compare_gan_variants gan_variants_reference.json
	python tools/stylegan_reference.py stylegan_reference.json
	$(MAKE) tests/compare_stylegan && ./tests/compare_stylegan stylegan_reference.json
	python tools/progressive_reference.py progressive_reference.json
	$(MAKE) tests/compare_progressive && ./tests/compare_progressive progressive_reference.json
	python tools/cyclegan_reference.py cyclegan_reference.json
	$(MAKE) tests/compare_cyclegan && ./tests/compare_cyclegan cyclegan_reference.json
	python tools/clip_reference.py clip_reference.json
	$(MAKE) tests/compare_clip && ./tests/compare_clip clip_reference.json
	python tools/train_sim_reference.py train_sim_reference.json
	$(MAKE) tests/compare_train_sim && ./tests/compare_train_sim train_sim_reference.json
	python tools/llama_reference.py llama_reference.json
	$(MAKE) tests/compare_llama && ./tests/compare_llama llama_reference.json
	python tools/bert_reference.py bert_reference.json
	$(MAKE) tests/compare_bert && ./tests/compare_bert bert_reference.json
	python tools/vit_reference.py vit_reference.json
	$(MAKE) tests/compare_vit && ./tests/compare_vit vit_reference.json
	python tools/electra_reference.py electra_reference.json
	$(MAKE) tests/compare_electra && ./tests/compare_electra electra_reference.json
	python tools/rwkv_reference.py rwkv_reference.json
	$(MAKE) tests/compare_rwkv && ./tests/compare_rwkv rwkv_reference.json
	python tools/retnet_reference.py retnet_reference.json
	$(MAKE) tests/compare_retnet && ./tests/compare_retnet retnet_reference.json
	python tools/xlstm_reference.py xlstm_reference.json
	$(MAKE) tests/compare_xlstm && ./tests/compare_xlstm xlstm_reference.json
	python tools/mamba_reference.py mamba_reference.json
	$(MAKE) tests/compare_mamba && ./tests/compare_mamba mamba_reference.json
	python tools/deberta_reference.py deberta_reference.json
	$(MAKE) tests/compare_deberta && ./tests/compare_deberta deberta_reference.json
	python tools/train_utils_reference.py train_utils_reference.json
	$(MAKE) tests/compare_train_utils && ./tests/compare_train_utils train_utils_reference.json
	python tools/rl_utils_reference.py rl_utils_reference.json
	$(MAKE) tests/compare_rl_utils && ./tests/compare_rl_utils rl_utils_reference.json
	python tools/sae_reference.py sae_reference.json
	$(MAKE) tests/compare_sae && ./tests/compare_sae sae_reference.json
	python tools/tree_attn_reference.py tree_attn_reference.json
	$(MAKE) tests/compare_tree_attn && ./tests/compare_tree_attn tree_attn_reference.json
	python tools/hallu_reference.py hallu_reference.json
	$(MAKE) tests/compare_hallu && ./tests/compare_hallu hallu_reference.json
	python tools/block_mgr_reference.py block_mgr_reference.json
	$(MAKE) tests/compare_block_mgr && ./tests/compare_block_mgr block_mgr_reference.json
	python tools/conv_reference.py conv_reference.json
	$(MAKE) tests/compare_conv && ./tests/compare_conv conv_reference.json
	python tools/llm_reference.py llm_reference.json
	$(MAKE) tests/compare_llm && ./tests/compare_llm llm_reference.json
	python tools/augment_reference.py augment_reference.json
	$(MAKE) tests/compare_augment && ./tests/compare_augment augment_reference.json
	python tools/ffn_reference.py ffn_reference.json
	$(MAKE) tests/compare_ffn && ./tests/compare_ffn ffn_reference.json
	python tools/adapt_reference.py adapt_reference.json
	$(MAKE) tests/compare_adapt && ./tests/compare_adapt adapt_reference.json
	python tools/heads_reference.py heads_reference.json
	$(MAKE) tests/compare_heads && ./tests/compare_heads heads_reference.json
	python tools/text_tok_reference.py text_tok_reference.json
	$(MAKE) tests/compare_text_tok && ./tests/compare_text_tok text_tok_reference.json

clean:
	rm -f $(BINS) $(BINS:=.exe) $(addprefix tests/,$(TESTS)) \
	      $(addsuffix .exe,$(addprefix tests/,$(TESTS))) *.o *.obj \
	      reference.txt c_model.json c_forward.txt py_model.json py_forward.txt self_model.json \
	      norm_reference.json conv_reference.json llm_reference.json \
	      adamw_reference.json lion_reference.json adafactor_reference.json \
	      sched_extra_reference.json rmsnorm_reference.json activations_reference.json \
	      loss_reference.json metrics_reference.json augment_reference.json ffn_reference.json adapt_reference.json heads_reference.json text_tok_reference.json text_metrics_reference.json audio_metrics_reference.json visual_metrics_reference.json ctc_beam_reference.json onnx_mlp_reference.json offline_rl_reference.json egnn_reference.json spec_decode_reference.json diffusion_reference.json flowmatching_reference.json tts_reference.json pix2pix_reference.json gan_variants_reference.json stylegan_reference.json progressive_reference.json cyclegan_reference.json clip_reference.json train_sim_reference.json llama_reference.json bert_reference.json vit_reference.json electra_reference.json rwkv_reference.json retnet_reference.json xlstm_reference.json mamba_reference.json deberta_reference.json train_utils_reference.json rl_utils_reference.json sae_reference.json tree_attn_reference.json hallu_reference.json block_mgr_reference.json

.PHONY: all test check clean
