#!/usr/bin/env python3
"""生成 CLIP 双塔（clip_loss / similarity / retrieval / CLIPProjector）的 C 对拍参考。

用法: python tools/clip_reference.py clip_reference.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import AI  # noqa: E402


def flat(x):
    if isinstance(x, (int, float)):
        return [float(x)]
    out = []
    for v in x:
        out.extend(flat(v))
    return out


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "clip_reference.json"

    proj = AI.CLIPProjector(image_dim=3, text_dim=4, embed_dim=5,
                            temperature=0.07, seed=7)
    image_feats = [[0.2, -0.4, 0.7], [0.1, 0.3, -0.2], [-0.5, 0.6, 0.9]]
    text_feats = [[0.3, -0.1, 0.5, 0.2], [-0.2, 0.4, 0.1, -0.3], [0.7, 0.2, -0.6, 0.4]]

    img_emb, txt_emb = proj.project(image_feats, text_feats)
    loss = proj.loss(image_feats, text_feats)
    loss2, grads = proj.loss_and_grads(image_feats, text_feats)
    before = {
        "W_img": flat(proj.W_img), "b_img": flat(proj.b_img),
        "W_txt": flat(proj.W_txt), "b_txt": flat(proj.b_txt),
    }
    lr = 0.05
    proj.apply_grads(grads, lr, kind="sgd")
    after = {
        "W_img": flat(proj.W_img), "b_img": flat(proj.b_img),
        "W_txt": flat(proj.W_txt), "b_txt": flat(proj.b_txt),
    }

    # 独立的相似度 / clip_loss / 检索用例
    sim_imgs = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [1.0, 1.0, 0.0]]
    sim_txts = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    sim = AI.clip_similarity(sim_imgs, sim_txts)

    cl_imgs = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    cl_txts = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    cl_loss, g_img, g_txt = AI.clip_loss(cl_imgs, cl_txts, temperature=0.07)
    acc = AI.contrastive_retrieval_accuracy(cl_imgs, cl_txts)

    payload = {
        "meta": {"source": "clip_reference.py", "tolerance": 1e-12},
        "param_count": proj.param_count(),
        "model": {"image_dim": 3, "text_dim": 4, "embed_dim": 5,
                  "temperature": 0.07, "seed": 7},
        "inputs": {"image_feats": flat(image_feats), "text_feats": flat(text_feats)},
        "weights": before,
        "project": {"img_emb": flat(img_emb), "txt_emb": flat(txt_emb)},
        "loss": loss,
        "loss2": loss2,
        "grads": {
            "W_img": flat(grads["W_img"]), "b_img": flat(grads["b_img"]),
            "W_txt": flat(grads["W_txt"]), "b_txt": flat(grads["b_txt"]),
        },
        "lr": lr,
        "weight_keys": list(before.keys()),
        "after": after,
        "similarity": {"imgs": flat(sim_imgs), "txts": flat(sim_txts), "sim": flat(sim),
                       "n_img": 3, "n_txt": 3, "d_img": 3},
        "clip_loss": {"imgs": flat(cl_imgs), "txts": flat(cl_txts), "loss": cl_loss,
                      "g_img": flat(g_img), "g_txt": flat(g_txt), "d": 2},
        "retrieval": {"imgs": flat(cl_imgs), "txts": flat(cl_txts),
                      "i2t": acc["i2t"], "t2i": acc["t2i"]},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print("wrote", out_path, "proj_loss", loss, "pc", proj.param_count())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
