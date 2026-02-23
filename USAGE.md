# EVA - RNA Sequence Generation and Scoring

EVA offers an open-source toolkit for RNA sequence generation and scoring. Inspired by the success of protein structure prediction and design tools, EVA brings similar capabilities to the RNA domain. Whether you need to generate novel RNA sequences from scratch, fill in missing regions of existing sequences, or evaluate how well a given sequence fits a particular biological context, EVA provides a flexible and powerful interface.

The system supports two primary modes: **Generation** (creating new sequences) and **Scoring** (evaluating existing sequences). Within Generation, there are two approaches: CLM (Causal Language Model) for autoregressive sequence generation, and GLM (General Language Model) for span infilling.

---

## Table of Contents

- [Running the Scripts](#running-the-scripts)
- [Generation - CLM](#generation---clm)
  - [Unconditional Generation](#unconditional-generation)
  - [Conditional Generation](#conditional-generation)
  - [Continuation Mode](#continuation-mode)
- [Generation - GLM](#generation---glm)
- [Scoring](#scoring)
  - [RNA Mode](#rna-mode)
  - [Protein Mode](#protein-mode)
- [Condition Control](#condition-control)
- [Sampling Parameters](#sampling-parameters)
- [Batch Processing with YAML](#batch-processing-with-yaml)
- [Input/Output Formats](#inputoutput-formats)
- [A Note on max_length](#a-note-on-max_length)

---

## Running the Scripts

EVA provides two main entry points: `generate.py` for sequence generation and `predict.py` for sequence scoring. These scripts accept parameters either directly from the command line or through YAML configuration files. For simple one-off tasks, command line arguments are often the quickest way to get started. For complex workflows involving multiple generation tasks or systematic evaluation, YAML configuration files provide a more organized approach.

The basic structure for running generation is:

```bash
python /eva/tools/generate.py [options]
```

And for scoring:

```bash
python /eva/tools/predict.py [options]
```

Let's walk through each major use case in detail.

---

## Generation - CLM

CLM (Causal Language Model) is the primary generation mode in EVA. It generates RNA sequences autoregressively from left to right, predicting one nucleotide at a time based on all previous nucleotides. This is similar to how language models generate text, but trained specifically on RNA sequences.

### Unconditional Generation

Let's first look at how you would do unconditional generation of RNA sequences. "Unconditional" means the model generates sequences without any guidance about what type of RNA it should be - the model will produce a diverse mixture of sequences based on what it learned during training.

For this, you simply need to specify:
1. The model checkpoint path
2. The number of sequences you want to generate
3. Where to save the output

**Command line:**

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --num_seqs 1000 \
    --output /output/unconditional.fa
```

What does each parameter mean?
- `--checkpoint` tells the system where to find the trained model weights
- `--format clm` specifies that we want to use the causal language model approach (this is also the default, but being explicit helps clarity)
- `--num_seqs` controls how many sequences to generate - here we're asking for 1000
- `--output` specifies where to write the resulting FASTA file

**YAML config:**

For batch processing or more complex scenarios, you might prefer a YAML configuration file:

```yaml
# config.yaml
checkpoint: /path/to/model
output_dir: ./output

tasks:
  - name: unconditional
    mode: generation
    format: clm
    num_seqs: 1000
```

The YAML approach becomes particularly useful when you want to run multiple generation tasks in sequence, or when you need to systematically explore different parameters.

### Conditional Generation

One of EVA's most powerful features is the ability to condition generation on specific biological context. You can specify the RNA type (such as mRNA, tRNA, etc.) and/or the species from which the sequences should be drawn. This is incredibly useful when you need sequences that are biologically meaningful for your specific research context.

#### Specify RNA Type Only

If you want to generate sequences that look like a specific type of RNA, you can use the `--rna_type` parameter. This tells the model to generate sequences that match the patterns it learned for that RNA type during training.

For example, to generate mRNA sequences:

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --rna_type mRNA \
    --num_seqs 1000 \
    --output /output/mrna.fa
```

The model has learned the characteristic patterns of different RNA types - for instance, mRNAs will have certain sequence features that distinguish them from tRNAs or rRNAs. By specifying the RNA type, you guide the generation toward these learned patterns.

**YAML config:**

```yaml
tasks:
  - name: mrna_generation
    mode: generation
    format: clm
    rna_type: mRNA
    num_seqs: 1000
```

#### Specify Species Only

You can also condition on species. The model has learned the sequence preferences of different organisms, so generating conditioned on a specific species will produce sequences that reflect those evolutionary patterns.

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --taxid 9606 \
    --num_seqs 1000 \
    --output /output/human.fa
```

Here, `--taxid 9606` specifies Homo sapiens (human). The model will generate sequences that reflect the codon usage patterns and sequence biases typical of human RNA.

You can also use the species name directly:

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --species homo_sapiens \
    --num_seqs 1000 \
    --output /output/human.fa
```

Or provide the full lineage string in Greengenes format:

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --lineage "D__Eukaryota;P__Chordata;C__Mammalia;O__Primates;F__Hominidae;G__Homo;S__Homo sapiens" \
    --num_seqs 1000 \
    --output /output/human.fa
```

**YAML config:**

```yaml
tasks:
  - name: human_generation
    mode: generation
    format: clm
    taxid: "9606"
    lineage: "D__Eukaryota;P__Chordata;C__Mammalia;O__Primates;F__Hominidae;G__Homo;S__Homo sapiens"
    num_seqs: 1000
```

#### Specify Both RNA Type and Species

For the most specific generation, you can combine both RNA type and species conditions. This produces sequences that should satisfy both criteria simultaneously.

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --rna_type mRNA \
    --taxid 9606 \
    --num_seqs 1000 \
    --output /output/human_mrna.fa
```

This will generate human mRNA sequences - sequences that look like both human RNA and mRNA.

**YAML config:**

```yaml
tasks:
  - name: human_mrna
    mode: generation
    format: clm
    rna_type: mRNA
    taxid: "9606"
    lineage: "D__Eukaryota;P__Chordata;C__Mammalia;O__Primates;F__Hominidae;G__Homo;S__Homo sapiens"
    num_seqs: 1000
```

### Continuation Mode

Sometimes you don't want to generate a completely new sequence - instead, you want to extend an existing one. Perhaps you have a partial sequence and want to see how the model would complete it, or you want to generate variants of a known sequence. The continuation mode is designed exactly for this purpose.

There are two directions available: forward continuation extends the 3' end of your sequence (generating what comes next), while reverse continuation extends the 5' end (generating what came before).

#### Forward Continuation (generate 3' end)

This keeps the 5' portion of your input sequence and generates what should come after. You need to specify how much of the original sequence to keep using either `--split_ratio` (as a fraction of total length) or `--split_pos` (as an exact position).

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

In this example, `--split_ratio 0.5` means "keep the first 50% of each input sequence and generate the remaining 50%". The `--num_seqs 5` parameter means "for each input sequence, generate 5 different completions".

**YAML config:**

```yaml
tasks:
  - name: forward_continue
    mode: generation
    format: clm
    input: ./input/partial_seq.fa
    direction: forward
    split_ratio: 0.5
    num_seqs: 5
    output_details: true
```

The `output_details: true` flag adds extra information to the output, including the original prompt, the ground truth (if you had a complete sequence), and the generated content.

#### Reverse Continuation (generate 5' end)

This keeps the 3' portion of your input sequence and generates what should come before it. This is useful when you have the end of a sequence and want to explore possible 5' beginnings.

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

Here, `--split_pos 699` means "keep everything from position 699 onward and generate everything before that position". The output sequence will be physically reversed (meaning the original 3' end becomes the 5' end of the output).

**YAML config:**

```yaml
tasks:
  - name: reverse_continue
    mode: generation
    format: clm
    input: ./input/partial_seq.fa
    direction: reverse
    split_pos: 699
    num_seqs: 20
```

---

## Generation - GLM

GLM (General Language Model) performs a different kind of generation task called "span infilling". Rather than generating a complete sequence from scratch, GLM takes an existing sequence, masks out a portion of it (called a "span"), and generates what should fill that gap based on the surrounding context.

This is particularly useful for:
- Completing partial sequences where you know the surrounding regions
- Exploring alternative sequences at specific positions
- Evaluating what the model thinks should go in a particular region

### Basic Span Infilling

To use GLM, you need to provide an input FASTA file with sequences. The system will then mask a span within each sequence and generate what should fill that span.

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format glm \
    --input /input/sequences.fa \
    --span_ratio 0.1 \
    --num_seqs 5 \
    --output /output/glm_output.fa
```

What do these GLM-specific parameters mean?
- `--format glm` tells the system to use the span infilling mode
- `--input` provides the sequences to work with (required for GLM)
- `--span_ratio 0.1` means "mask 10% of each sequence's length as the span"
- `--num_seqs 5` means "generate 5 different fillings for each span"

**YAML config:**

```yaml
tasks:
  - name: glm_infill
    mode: generation
    format: glm
    input: ./input/sequences.fa
    span_ratio: 0.1
    span_position: random
    span_id: random
    num_seqs: 5
```

### Controlling Span Parameters

GLM offers several parameters to control exactly how the span is selected and filled:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--span_length` | Fixed number of nucleotides to mask | `--span_length 20` |
| `--span_ratio` | Fraction of sequence to mask | `--span_ratio 0.1` |
| `--span_position` | Where to place the span: "random" or specific index | `--span_position 100` |
| `--span_id` | Which span token to use: "random" or 0-49 | `--span_id 0` |

**Using fixed span length:**

If you want precise control over how much is masked, use `--span_length`:

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format glm \
    --input /input/sequences.fa \
    --span_length 20 \
    --span_position 0 \
    --span_id 0 \
    --num_seqs 5 \
    --output /output/glm_fixed.fa
```

This masks exactly 20 nucleotides starting at position 0 (the beginning of each sequence).

**YAML config:**

```yaml
tasks:
  - name: glm_fixed_span
    mode: generation
    format: glm
    input: ./input/sequences.fa
    span_length: 20
    span_position: 0
    span_id: 0
    num_seqs: 5
```

### GLM with Conditions

You can also apply biological conditions (RNA type, species) during span infilling:

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format glm \
    --input /input/sequences.fa \
    --rna_type mRNA \
    --taxid 9606 \
    --span_ratio 0.2 \
    --num_seqs 5 \
    --temperature 0.7 \
    --output /output/glm_condition.fa
```

This will generate infills that are consistent with both the local sequence context AND the specified biological conditions.

**YAML config:**

```yaml
tasks:
  - name: glm_human_mrna
    mode: generation
    format: glm
    input: ./input/sequences.fa
    rna_type: mRNA
    taxid: "9606"
    lineage: "D__Eukaryota;P__Chordata;..."
    span_ratio: 0.2
    num_seqs: 5
    temperature: 0.7
```

---

## Scoring

Scoring allows you to evaluate how well a given RNA sequence fits the model's learned distribution. In practical terms, this means computing the log-likelihood of a sequence - how probable is this sequence according to what the model has learned? Higher scores indicate that the sequence is more "model-like", while lower scores suggest the sequence is unusual or atypical.

This is useful for:
- Evaluating designed sequences - are they reasonable?
- Comparing variants - which variant is more likely?
- Filtering generated sequences - keep only the best ones
- Analyzing mutations - how does a mutation affect sequence probability?

### RNA Mode

In RNA mode, sequences are scored directly as RNA sequences. This is the appropriate mode when you have RNA sequences and want to evaluate them as-is.

**Command line:**

```bash
python /eva/tools/predict.py \
    --checkpoint /path/to/model \
    --input /input/sequences.fa \
    --output /output/scores.json
```

What does each parameter mean?
- `--checkpoint` points to the model to use for scoring
- `--input` provides the sequences to score
- `--output` specifies where to write the results (in JSON format)

**YAML config:**

```yaml
# config_score.yaml
checkpoint: /path/to/model
output_dir: ./scores

tasks:
  - name: score_sequences
    mode: scoring
    input: ./input/sequences.fa
    output: ./scores/sequences.json
```

#### Scoring with Conditions

You can also score sequences conditioned on specific RNA type and/or species. This evaluates how well the sequence fits both its own patterns AND the specified biological context.

```bash
python /eva/tools/predict.py \
    --checkpoint /path/to/model \
    --input /input/sequences.fa \
    --output /output/scores.json \
    --rna_type mRNA \
    --taxid 9606
```

This is useful when you want to evaluate whether sequences are good examples of a specific RNA type from a specific organism.

**YAML config:**

```yaml
tasks:
  - name: score_human_mrna
    mode: scoring
    input: ./input/sequences.fa
    output: ./scores/human_mrna.json
    rna_type: mRNA
    taxid: "9606"
    lineage: "D__Eukaryota;P__Chordata;..."
```

#### Normalization Options

EVA provides several options to normalize the scores, which can be important when comparing sequences of different lengths:

| Option | Description |
|--------|-------------|
| `--normalize` | Apply token-level mean normalization - divides by the number of tokens |
| `--exclude_special_tokens` | Exclude 5/3/\<eos\> tokens from the calculation |
| `--length_normalize` | Divide the final score by the sequence length |

These options can be combined:

```bash
python /eva/tools/predict.py \
    --checkpoint /path/to/model \
    --input /input/sequences.fa \
    --output /output/scores.json \
    --normalize \
    --exclude_special_tokens \
    --length_normalize
```

**YAML config:**

```yaml
tasks:
  - name: score_normalized
    mode: scoring
    input: ./input/sequences.fa
    output: ./scores/normalized.json
    normalize: true
    exclude_special_tokens: true
    length_normalize: true
```

### Protein Mode

Protein mode is a specialized scoring mode for protein sequences. Since the model is trained on RNA, protein sequences must first be "reverse translated" to RNA before they can be scored. This uses a codon table to convert amino acids back to nucleotides.

This is particularly useful when:
- You have designed protein sequences and want to evaluate them
- You're interested in the coding potential of sequences
- You want to score proteins using RNA-based metrics

**Command line:**

```bash
python /eva/tools/predict.py \
    --checkpoint /path/to/model \
    --input /input/proteins.fa \
    --output /output/protein_scores.json \
    --mode protein \
    --codon_optimization first
```

What does `--codon_optimization` mean? Since multiple codons can code for the same amino acid (for example, both AUG and GUG code for methionine), we need to decide which RNA sequence to use when converting a protein sequence. Two strategies are available:
- `first`: Use the first codon in the codon table for each amino acid
- `most_frequent`: Use the codon that appears most frequently in the specified species

**YAML config:**

```yaml
tasks:
  - name: score_protein
    mode: scoring
    input: ./input/proteins.fa
    output: ./scores/protein.json
    scoring_mode: protein
    codon_optimization: first
```

---

## Condition Control

### RNA Types

EVA supports conditioning generation and scoring on 15 different RNA types. Each RNA type has characteristic sequence patterns that the model has learned:

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
| ribozyme | Catalytic RNA - has enzymatic activity |
| scaRNA | Small Cajal body RNA - modifies snRNAs |
| Y_RNA | Y RNA - involved in RNA quality control |
| vault_RNA | Vault RNA - part of the vault ribonucleoprotein complex |

To use any of these, simply specify `--rna_type` with the appropriate value:

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --rna_type tRNA \
    --num_seqs 100 \
    --output /output/trna.fa
```

### Species/Lineage

EVA can condition on specific species using either TaxID, species name, or full lineage string. The built-in species database includes:

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

You can use TaxID (recommended for simplicity):

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --taxid 10090 \
    --num_seqs 100 \
    --output /output/mouse.fa
```

Or species name:

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --species homo_sapiens \
    --num_seqs 100 \
    --output /output/human.fa
```

Or the complete lineage string in Greengenes format:

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --lineage "D__Eukaryota;P__Chordata;C__Mammalia;O__Primates;F__Hominidae;G__Homo;S__Homo sapiens" \
    --num_seqs 100 \
    --output /output/human_lineage.fa
```

---

## Sampling Parameters

When generating sequences, you have control over the sampling process through several parameters. These affect the randomness and diversity of the generated sequences.

| Parameter | Description | Recommended Range |
|-----------|-------------|------------------|
| `--temperature` | Controls randomness - lower values make output more deterministic, higher values add more randomness | 0.1 - 1.5 |
| `--top_k` | Only consider the top k most likely nucleotides at each position | 10 - 100 |
| `--top_p` | Nucleus sampling - consider only the smallest set of nucleotides whose cumulative probability exceeds p | 0.8 - 0.95 |

### Temperature

The temperature parameter is perhaps the most important one to understand. A temperature of 1.0 uses the model's unmodified probability distribution. Lower temperatures (e.g., 0.5) make the model more confident - it will choose high-probability nucleotides more consistently, producing more "typical" sequences. Higher temperatures (e.g., 1.5) flatten the distribution, making even unlikely nucleotides more likely to be chosen, producing more diverse and potentially novel sequences.

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --temperature 0.5 \
    --top_k 20 \
    --num_seqs 100 \
    --output /output/deterministic.fa
```

This produces more deterministic, "typical" sequences.

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --temperature 1.2 \
    --top_k 100 \
    --num_seqs 100 \
    --output /output/diverse.fa
```

This produces more diverse sequences with more variation.

### Top-k and Top-p

Top-k sampling limits consideration to only the k most likely nucleotides at each step. Top-p (also called "nucleus sampling") takes a different approach - it selects the smallest set of nucleotides whose cumulative probability exceeds a threshold p.

These can be used together or separately:

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --temperature 0.8 \
    --top_p 0.9 \
    --num_seqs 100 \
    --output /output/nucleus.fa
```

---

## Batch Processing with YAML

For complex workflows involving multiple tasks, YAML configuration files provide a powerful and organized approach. Instead of running many individual commands, you can define all your tasks in a single configuration file and run them together.

### Complete Configuration Example

```yaml
# config.yaml
checkpoint: /path/to/model
output_dir: ./output

defaults:
  temperature: 1.0
  top_k: 50
  max_length: 8192
  batch_size: 1
  min_length: 10

tasks:
  # Unconditional generation
  - name: unconditional
    mode: generation
    format: clm
    num_seqs: 1000

  # Conditional generation - human mRNA
  - name: human_mrna
    mode: generation
    format: clm
    rna_type: mRNA
    taxid: "9606"
    lineage: "D__Eukaryota;P__Chordata;C__Mammalia;O__Primates;F__Hominidae;G__Homo;S__Homo sapiens"
    num_seqs: 1000

  # Conditional generation - mouse tRNA
  - name: mouse_trna
    mode: generation
    format: clm
    rna_type: tRNA
    taxid: "10090"
    lineage: "D__Eukaryota;P__Chordata;C__Mammalia;O__Rodentia;F__Muridae;G__Mus;S__Mus musculus"
    num_seqs: 500
    temperature: 0.7

  # Forward continuation
  - name: forward_continue
    mode: generation
    format: clm
    input: ./input/seqs.fa
    direction: forward
    split_ratio: 0.5
    num_seqs: 10

  # GLM span infilling
  - name: glm_infill
    mode: generation
    format: glm
    input: ./input/seqs.fa
    span_ratio: 0.1
    num_seqs: 5
```

The `defaults` section lets you specify parameters that apply to all tasks, which can then be overridden in individual tasks as needed. In this example, all tasks use temperature 1.0 by default, but the mouse_tRNA task overrides it to 0.7.

### Scoring Configuration Example

```yaml
# config_score.yaml
checkpoint: /path/to/model
output_dir: ./scores

defaults:
  normalize: false
  exclude_special_tokens: false
  length_normalize: false
  batch_size: 128

tasks:
  # Basic scoring
  - name: score_basic
    mode: scoring
    input: ./input/seqs.fa
    output: ./scores/basic.json

  # Score with RNA type condition
  - name: score_lncrna
    mode: scoring
    input: ./input/seqs.fa
    output: ./scores/lncrna.json
    rna_type: lncRNA
    normalize: true

  # Score with species condition
  - name: score_human
    mode: scoring
    input: ./input/seqs.fa
    output: ./scores/human.json
    taxid: "9606"

  # Full normalization options
  - name: score_full_norm
    mode: scoring
    input: ./input/seqs.fa
    output: ./scores/full_norm.json
    rna_type: mRNA
    taxid: "9606"
    normalize: true
    exclude_special_tokens: true
    length_normalize: true

  # Protein mode
  - name: score_protein
    mode: scoring
    input: ./input/proteins.fa
    output: ./scores/protein.json
    scoring_mode: protein
    codon_optimization: first
```

### Running Batch Tasks

Once you've created your configuration file, running the tasks is straightforward:

**Run all tasks:**

```bash
python /eva/tools/generate.py --config /path/to/config.yaml
```

**Run a specific task:**

```bash
python /eva/tools/generate.py --config /path/to/config.yaml --task human_mrna
```

**Check task status:**

```bash
python /eva/tools/generate.py --config /path/to/config.yaml --status
```

**Override device:**

```bash
python /eva/tools/generate.py --config /path/to/config.yaml --device cuda:1
```

---

## Input/Output Formats

### Input FASTA Format

EVA expects input sequences in standard FASTA format:

```
>sequence_id_1
AUGCGCUAUGCGCUAUGCGCUAUGCGCUAUGCGCUAUGCGCUAUGCGCU
>sequence_id_2
AUGAAAAUGCGGCCGCAUUACGUAAACGGCCGCAAAUGUUUCCGGCAAA
```

Each sequence consists of a header line (starting with `>`) followed by the sequence on one or more lines. For GLM and continuation modes, provide full RNA sequences.

### Output FASTA (Generation)

Generated sequences are saved in FASTA format:

```
>unconditional_0
AUGCGCUAUGCGCUAUGCGCUAUGCGCUAUGCGCUAUGCGCUAUGCGCU
>human_mrna_0
AUGAAAAUGCGGCCGCAUUACGUAAACGGCCGCAAAUGUUUCCGGCAAA
```

### Output Details (GLM / Continuation)

When `--output_details` is enabled, additional information is included:

```
>test_seq_sample0_forward_split50
PROMPT: AUGCGCUAUGCGCUAUGCG
GROUND_TRUTH: CU AUGCGCUAUGCG
GENERATED: CU AAUGCGCUAGCG
FULL_SEQ: AUGCGCUAUGCGCUAAUGCGCUAGCG
```

This shows:
- `PROMPT`: The input sequence that was kept
- `GROUND_TRUTH`: The original content that was masked (if known)
- `GENERATED`: What the model generated to fill the span
- `FULL_SEQ`: The complete reconstructed sequence

### Output JSON (Scoring)

Scoring results are saved in JSON format:

```json
{
  "input_file": "/path/to/sequences.fa",
  "checkpoint": "/path/to/model",
  "mode": "scoring",
  "normalize": false,
  "exclude_special_tokens": false,
  "length_normalize": false,
  "num_sequences": 3,
  "condition": {
    "rna_type": "mRNA",
    "taxid": "9606",
    "species": null,
    "lineage": "D__Eukaryota;..."
  },
  "scores": [
    {
      "index": 0,
      "header": "seq1",
      "sequence": "AUGGCCGUAGU...",
      "length": 67,
      "log_likelihood": -56.25
    },
    {
      "index": 1,
      "header": "seq2",
      "sequence": "AUGCGCUAUGC...",
      "length": 63,
      "log_likelihood": -43.23
    }
  ]
}
```

The key field is `log_likelihood` - higher (less negative) values indicate more probable sequences.

---

## A Note on max_length

There's one important parameter worth discussing in detail: `--max_length`.

This parameter controls the maximum length of generated sequences. The default value is 8192 nucleotides, which is quite long. While this provides flexibility for generating very long sequences, it also means generation can take a very long time for most practical use cases.

**We recommend** using a much smaller value for most tasks. For initial exploration and testing, values like 100-500 are typically sufficient:

```bash
python /eva/tools/generate.py \
    --checkpoint /path/to/model \
    --format clm \
    --max_length 200 \
    --num_seqs 100 \
    --output /output/test.fa
```

You can then increase this value for production runs once you've verified everything works correctly.

---

We hope this documentation helps you get the most out of EVA. The system is designed to be flexible and powerful while remaining accessible to researchers who want to focus on their biological questions rather than the technical details.

Now, let's go design some RNA. Have fun!

- The EVA Team
