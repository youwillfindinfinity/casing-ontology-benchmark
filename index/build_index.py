#!/usr/bin/env python3
"""
build_index.py — build a FAISS IndexFlatIP over the NCBI Taxonomy canonical names
for one sentence-transformer model, exactly as used in the paper.

The index performs *exact* (exhaustive) cosine-similarity search via inner product on
L2-normalised embeddings, eliminating approximate-nearest-neighbour error as a confound
so the reported Hits@1 is an upper performance bound.

Only entries of name type `scientific name` are indexed (2.69M rows); synonym / common /
equivalent / genbank-common names are excluded so that synonym_setB genuinely tests
cross-form retrieval.

Usage:
    python index/build_index.py --model all-MiniLM-L6-v2 \
        --names-dmp data/names.dmp --out index/all-MiniLM-L6-v2.faiss

Requires: sentence-transformers, faiss-cpu (or faiss-gpu), numpy, tqdm.
GPU strongly recommended (~2–8 h/model on an A100; peak VRAM 2–14 GB).
"""
import argparse
import os
import sys

import numpy as np


def load_scientific_names(names_dmp):
    """Yield canonical scientific names from NCBI Taxonomy names.dmp.

    names.dmp format (tab-pipe-tab separated):
        tax_id | name_txt | unique_name | name_class |
    We keep rows whose name_class == 'scientific name'.
    """
    names, taxids = [], []
    with open(names_dmp, encoding="utf-8") as fh:
        for line in fh:
            parts = [p.strip() for p in line.rstrip("\n").rstrip("|").split("|")]
            if len(parts) < 4:
                continue
            tax_id, name_txt, _unique, name_class = parts[0], parts[1], parts[2], parts[3]
            if name_class == "scientific name":
                taxids.append(int(tax_id))
                names.append(name_txt)
    return names, np.asarray(taxids, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True,
                    help="HuggingFace sentence-transformer id or local path")
    ap.add_argument("--names-dmp", required=True, help="path to NCBI Taxonomy names.dmp")
    ap.add_argument("--out", required=True, help="output .faiss index path")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    args = ap.parse_args()

    try:
        import faiss
        from sentence_transformers import SentenceTransformer
        from tqdm import tqdm
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}. Run `pip install -r requirements.txt`.")

    print(f"[1/4] Loading scientific names from {args.names_dmp} ...")
    names, taxids = load_scientific_names(args.names_dmp)
    print(f"      {len(names):,} canonical names loaded")

    print(f"[2/4] Loading model {args.model} ...")
    model = SentenceTransformer(args.model, device=args.device)
    dim = model.get_sentence_embedding_dimension()

    print(f"[3/4] Embedding (dim={dim}, batch={args.batch_size}) ...")
    index = faiss.IndexFlatIP(dim)               # exact inner-product (cosine on L2-norm)
    for start in tqdm(range(0, len(names), args.batch_size)):
        batch = names[start:start + args.batch_size]
        emb = model.encode(batch, batch_size=args.batch_size,
                           normalize_embeddings=True, show_progress_bar=False)
        index.add(np.asarray(emb, dtype=np.float32))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    print(f"[4/4] Writing index → {args.out} and taxid sidecar ...")
    faiss.write_index(index, args.out)
    np.save(args.out + ".taxids.npy", taxids)    # row i of index ↔ taxids[i]
    with open(args.out + ".meta", "w") as fh:
        fh.write(f"model\t{args.model}\ndim\t{dim}\nn_names\t{len(names)}\n"
                 f"name_class\tscientific name\nmetric\tinner_product_L2normalised\n")
    print(f"Done. {index.ntotal:,} vectors indexed.")


if __name__ == "__main__":
    main()
