# EVA: A Generative Foundation Model for Universal RNA Modeling and Design

<p align="center">
  <img src="fig/poster.svg" alt="RNAVerse" width="600">
</p>

**EVA** (Evolutionary Versatile Architect) is a generative RNA foundation model trained on **RNAVerse v1**, a curated atlas of 114 million full-length RNA sequences spanning all domains of life. Built on a 1.4B-parameter decoder-only Transformer with a Mixture-of-Experts (MoE) backbone and an 8,192-token context window, EVA unifies RNA sequence scoring and controllable design within a single framework.

### Key Features

- **Zero-shot fitness prediction** across RNA, DNA gene regions, and proteins via evolutionary likelihood scoring.
- **Controllable generation** across 11 RNA classes (mRNA, lncRNA, circRNA, tRNA, rRNA, miRNA, piRNA, sRNA, snRNA, snoRNA, viral RNA) conditioned on RNA type and taxonomic lineage. — no task-specific fine-tuning required
- **Two generation modes**: autoregressive CLM for de novo sequence design, and GLM (masked infilling) for targeted region redesign.
- **Fine-tuning** support is coming soon.

For instructions, details, and examples, please refer to our [technical report](TODO).

For checkpoints, please refer to our [Hugging Face](https://huggingface.co/yanjiehuang/EVA1).

---

## Table of Contents

- [Running the Scripts](#running-the-scripts)
- [Generation - CLM](#generation---clm)
  - [Unconditional Generation](#unconditional-generation)
  - [Conditional Generation](#conditional-generation)
  - [Continuation Mode](#continuation-mode)
- [Generation - GLM](#generation---glm)
  - [Unconditional Infilling](#unconditional-infilling)
  - [Conditional Infilling](#conditional-infilling)
- [Scoring](#scoring)
  - [RNA Mode](#rna-mode)
  - [Protein Mode](#protein-mode)
- [Condition Control](#condition-control)
- [Sampling Parameters](#sampling-parameters)
- [Batch Processing with YAML](#batch-processing-with-yaml)
- [Input/Output Formats](#inputoutput-formats)
- [Data Availability](#data-availability)

---

## Running the Scripts

EVA provides two main entry points:

```bash
python /eva/tools/generate.py [options]   # Sequence generation
python /eva/tools/predict.py [options]     # Sequence scoring
```

Parameters can be passed via command line or YAML configuration files (see [Batch Processing with YAML](#batch-processing-with-yaml)).

---

## Generation - CLM

CLM (Causal Language Model) generates RNA sequences autoregressively from left to right. This is the primary generation mode in EVA.

### Unconditional Generation

Generate sequences without any biological constraints:

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --num_seqs 1000 \
    --output /output/unconditional.fa
```

### Conditional Generation

EVA supports conditioning on **RNA type**, **species** (via TaxID, species name, or lineage string), or both. See [Condition Control](#condition-control) for the full list of supported RNA types and species.

```bash
# RNA type only
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --rna_type mRNA \
    --num_seqs 1000 \
    --output /output/mrna.fa

# Species only (via TaxID)
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --taxid 9606 \
    --num_seqs 1000 \
    --output /output/human.fa

# Both RNA type and species
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --rna_type mRNA \
    --taxid 9606 \
    --num_seqs 1000 \
    --output /output/human_mrna.fa
```

Species can also be specified via `--species homo_sapiens` or `--lineage "D__Eukaryota;P__Chordata;..."` in Greengenes format.

### Continuation Mode

Extend existing sequences in either direction. Use `--split_ratio` (fraction) or `--split_pos` (exact position) to control the split point.

**Forward** (extend 3' end):

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --input /input/partial_seq.fa \
    --direction forward \
    --split_ratio 0.5 \
    --num_seqs 5 \
    --output /output/continuation.fa
```

**Reverse** (extend 5' end):

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --input /input/partial_seq.fa \
    --direction reverse \
    --split_pos 699 \
    --num_seqs 20 \
    --output /output/reverse_continuation.fa
```

Add `--output_details` to include prompt, ground truth, and generated content in the output.

---

## Generation - GLM

GLM (General Language Model) performs span infilling — it masks a region within an existing sequence and generates what should fill the gap based on surrounding context. Like CLM, GLM supports both unconditional and conditional generation.

### Unconditional Infilling

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format glm \
    --input /input/sequences.fa \
    --span_ratio 0.1 \
    --num_seqs 5 \
    --output /output/glm_output.fa
```

### Conditional Infilling

Condition on RNA type and/or species to generate biologically consistent infills:

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format glm \
    --input /input/sequences.fa \
    --rna_type mRNA \
    --taxid 9606 \
    --span_ratio 0.2 \
    --num_seqs 5 \
    --output /output/glm_conditional.fa
```

### Span Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--span_length` | Fixed number of nucleotides to mask | `--span_length 20` |
| `--span_ratio` | Fraction of sequence to mask | `--span_ratio 0.1` |
| `--span_position` | Where to place the span: `random` or specific index | `--span_position 100` |
| `--span_id` | Which span token to use: `random` or 0-49 | `--span_id 0` |

---

## Scoring

Evaluate how well a given sequence fits the model's learned distribution by computing its log-likelihood. Higher (less negative) scores indicate more probable sequences.

### RNA Mode

```bash
python /eva/tools/predict.py \
    --checkpoint /path/to/model \
    --input /input/sequences.fa \
    --output /output/scores.json
```

Supports `--rna_type` and `--taxid` conditioning, same as generation.

### Protein Mode

Score protein sequences by reverse-translating them to RNA first:

```bash
python /eva/tools/predict.py \
    --checkpoint /path/to/model \
    --input /input/proteins.fa \
    --output /output/protein_scores.json \
    --mode protein \
    --codon_optimization first
```

`--codon_optimization` options: `first` (first codon in table) or `most_frequent` (most common codon for the species).

---

## Condition Control

### RNA Types

| RNA Type | Description |
|----------|-------------|
| mRNA | Messenger RNA - carries genetic information from DNA to ribosomes |
| tRNA | Transfer RNA - brings amino acids to the ribosome during translation |
| rRNA | Ribosomal RNA - forms the core of the ribosome structure |
| miRNA | MicroRNA - regulates gene expression |
| lncRNA | Long non-coding RNA - various regulatory functions |
| circRNA | Circular RNA - circularized RNA molecules |
| snoRNA | Small nucleolar RNA - modifies other RNAs |
| snRNA | Small nuclear RNA - involved in splicing |
| piRNA | PIWI-interacting RNA - silences transposons |
| sRNA | Small RNA - general category for small RNA molecules |
| viral_RNA | RNA from viruses |

### Species/Lineage

Species can be specified in three ways: `--taxid`, `--species`, or `--lineage` (Greengenes format).

Common species:

| TaxID | Species |
|-------|---------|
| 9606 | Homo sapiens (Human) |
| 10090 | Mus musculus (Mouse) |
| 10116 | Rattus norvegicus (Rat) |
| 7227 | Drosophila melanogaster (Fruit fly) |
| 6239 | Caenorhabditis elegans (Nematode) |
| 3702 | Arabidopsis thaliana (Plant) |
| 4932 | Saccharomyces cerevisiae (Yeast) |
| 562 | Escherichia coli (Bacteria) |

---

## Sampling Parameters

| Parameter | Description | Recommended Range |
|-----------|-------------|------------------|
| `--temperature` | Controls randomness. Lower = more deterministic, higher = more diverse | 0.1 - 1.5 |
| `--top_k` | Only consider the top k most likely nucleotides at each position | 10 - 100 |
| `--top_p` | Nucleus sampling — consider smallest set of nucleotides whose cumulative probability exceeds p | 0.8 - 0.95 |

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.9 \
    --num_seqs 100 \
    --output /output/sampled.fa
```

---

## Batch Processing with YAML

Define multiple tasks in a single YAML config file. The `defaults` section sets shared parameters, which individual tasks can override.

### Generation Config Example

```yaml
checkpoint: /path/to/model
output_dir: ./output

defaults:
  temperature: 1.0
  top_k: 50
  max_length: 8192
  batch_size: 1

tasks:
  - name: unconditional
    mode: generation
    format: clm
    num_seqs: 1000

  - name: human_mrna
    mode: generation
    format: clm
    rna_type: mRNA
    taxid: "9606"
    lineage: "D__Eukaryota;P__Chordata;C__Mammalia;O__Primates;F__Hominidae;G__Homo;S__Homo sapiens"
    num_seqs: 1000

  - name: glm_infill
    mode: generation
    format: glm
    input: ./input/seqs.fa
    span_ratio: 0.1
    num_seqs: 5
```

### Scoring Config Example

```yaml
checkpoint: /path/to/model
output_dir: ./scores

defaults:
  batch_size: 128

tasks:
  - name: score_basic
    mode: scoring
    input: ./input/seqs.fa

  - name: score_human_mrna
    mode: scoring
    input: ./input/seqs.fa
    rna_type: mRNA
    taxid: "9606"
    normalize: true
    exclude_special_tokens: true

  - name: score_protein
    mode: scoring
    input: ./input/proteins.fa
    scoring_mode: protein
    codon_optimization: first
```

### Running

```bash
python /eva/tools/generate.py --config config.yaml              # Run all tasks
python /eva/tools/generate.py --config config.yaml --task name   # Run specific task
python /eva/tools/generate.py --config config.yaml --device cuda:1  # Override device
```

---

## Input/Output Formats

### Input — FASTA

```
>sequence_id_1
AUGCGCUAUGCGCUAUGCGCUAUGCGCUAUGCGCUAUGCGCUAUGCGCU
>sequence_id_2
AUGAAAAUGCGGCCGCAUUACGUAAACGGCCGCAAAUGUUUCCGGCAAA
```

### Output — Generation (FASTA)

```
>unconditional_0
AUGCGCUAUGCGCUAUGCGCUAUGCGCUAUGCGCUAUGCGCUAUGCGCU
```

With `--output_details` (GLM / Continuation):

```
>test_seq_sample0_forward_split50
PROMPT: AUGCGCUAUGCGCUAUGCG
GROUND_TRUTH: CU AUGCGCUAUGCG
GENERATED: CU AAUGCGCUAGCG
FULL_SEQ: AUGCGCUAUGCGCUAAUGCGCUAGCG
```

### Output — Scoring (JSON)

```json
{
  "scores": [
    {
      "header": "seq1",
      "sequence": "AUGGCCGUAGU...",
      "length": 67,
      "log_likelihood": -1.25
    }
  ]
}
```

Higher (less negative) `log_likelihood` = better sequence.

---

## Data Availability

Some large files are not included in this repository due to size constraints. The following data can be downloaded from Zenodo:

- `checkpoint/` — Model checkpoints
- `eva_latest.tar` — Pre-built EVA Docker image
- `notebooks/interpretability_analysis/intermediate_data/*.npz` — Precomputed activation data
- `notebooks/tools/visualization/UMAP/taxid_phylum_mapping.json` — Taxonomy mapping data
