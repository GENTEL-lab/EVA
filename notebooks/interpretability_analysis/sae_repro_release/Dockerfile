# EVA SAE reproducibility image
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/EVA1

COPY notebooks/interpretability_analysis/sae_repro_release/requirements-sae-training.txt /tmp/requirements-sae-training.txt
RUN pip install --no-cache-dir -r /tmp/requirements-sae-training.txt && rm /tmp/requirements-sae-training.txt

COPY eva /workspace/EVA1/eva
COPY notebooks/interpretability_analysis/sae_repro_release /workspace/EVA1/notebooks/interpretability_analysis/sae_repro_release

RUN chmod +x /workspace/EVA1/notebooks/interpretability_analysis/sae_repro_release/scripts/*.sh \
    && chmod +x /workspace/EVA1/notebooks/interpretability_analysis/sae_repro_release/scripts/*.py

ENTRYPOINT ["/bin/bash"]
