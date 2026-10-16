# AI_C/c —— Python 版神经网络工具箱的 C 移植
#
# 如果你的环境没有 make，直接编译即可（Windows/MinGW 与 Linux/macOS 通用）：
#   gcc -std=c17 -O2 -Wall -Wextra -o demo demo.c ai.c tensor.c layers_conv.c \
#       layers_norm.c rnn.c attention.c containers.c json.c model_io.c text.c sched.c -lm
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
	python tools/conv_reference.py conv_reference.json
	$(MAKE) tests/compare_conv && ./tests/compare_conv conv_reference.json

clean:
	rm -f $(BINS) $(BINS:=.exe) $(addprefix tests/,$(TESTS)) \
	      $(addsuffix .exe,$(addprefix tests/,$(TESTS))) *.o *.obj \
	      reference.txt c_model.json c_forward.txt py_model.json py_forward.txt self_model.json \
	      norm_reference.json conv_reference.json

.PHONY: all test check clean
