"""
Generation Conditions

Encapsulates various conditions for RNA generation, supporting combinations of RNA types and species conditions.
Responsible for building prompts that conform to the original script format.

Strict reference: Prompt format from lines 53-68 of ORIGINAL_SCRIPT_ANALYSIS.md
"""

from dataclasses import dataclass
from typing import Optional

from .rna_types import validate_rna_type, get_rna_token
from .lineage import LineageDatabase


@dataclass
class GenerationCondition:
    """
    Generation Condition

    Encapsulates various conditions for RNA generation, supporting combinations of RNA types and species conditions.

    Attributes:
        rna_type: RNA type, such as "mRNA", "tRNA", etc.
        species: Species name, such as "homo_sapiens"
        taxid: NCBI Taxonomy ID, such as "9606"
        lineage: Complete Greengenes format lineage string (for advanced users)

    Priority: lineage > taxid > species

    Prompt format (reference lines 53-68 of ORIGINAL_SCRIPT_ANALYSIS.md):
        - Unconditional: <bos>5
        - RNA type only: <bos>|<rna_mRNA>|5
        - Species only: <bos>|d__eukaryota;...;s__homo_sapiens|5
        - RNA type + species: <bos>|d__...;s__homo_sapiens;<rna_mRNA>|5

    Example:
        # Specify RNA type only
        cond1 = GenerationCondition(rna_type="mRNA")

        # Specify species (by name)
        cond2 = GenerationCondition(species="homo_sapiens")

        # Specify species (by TaxID)
        cond3 = GenerationCondition(taxid="9606")

        # Combined conditions
        cond4 = GenerationCondition(rna_type="mRNA", taxid="9606")
    """

    rna_type: Optional[str] = None
    species: Optional[str] = None
    taxid: Optional[str] = None
    lineage: Optional[str] = None

    def validate(self) -> None:
        """
        Validate the validity of conditions

        Raises:
            ValueError: If the condition is invalid (e.g., unsupported RNA type)
        """
        if self.rna_type is not None and not validate_rna_type(self.rna_type):
            from .rna_types import RNA_TYPES
            raise ValueError(
                f"Unsupported RNA type: {self.rna_type}\n"
                f"Supported types: {list(RNA_TYPES.keys())}"
            )

    def resolve_lineage(self, lineage_db: LineageDatabase) -> Optional[str]:
        """
        Resolve and return the final lineage string

        Priority: lineage > taxid > species

        Args:
            lineage_db: Lineage database instance

        Returns:
            Greengenes format lineage string, or None if no species condition
        """
        # Directly specified lineage takes priority
        if self.lineage is not None:
            return self.lineage

        # Lookup by taxid
        if self.taxid is not None:
            lineage = lineage_db.get_lineage(taxid=self.taxid)
            if lineage is not None:
                return lineage

        # Lookup by species name
        if self.species is not None:
            lineage = lineage_db.get_lineage(species=self.species)
            if lineage is not None:
                return lineage

        return None

    def build_clm_prompt(self, lineage_db: Optional[LineageDatabase] = None) -> str:
        """
        Build CLM generation prompt

        Strict reference to format from lines 53-68 of ORIGINAL_SCRIPT_ANALYSIS.md:
            - Unconditional: <bos>5
            - RNA type only: <bos>|<rna_mRNA>|5
            - Species only: <bos>|d__eukaryota;...;s__homo_sapiens|5
            - RNA type + species: <bos>|d__...;s__homo_sapiens;<rna_mRNA>|5

        Args:
            lineage_db: Lineage database instance (if species condition needs to be resolved)

        Returns:
            CLM prompt string
        """
        # Validate conditions
        self.validate()

        # Resolve lineage
        # If lineage is directly specified, use it; otherwise resolve through lineage_db
        resolved_lineage = None
        if self.lineage is not None:
            resolved_lineage = self.lineage
        elif lineage_db is not None:
            resolved_lineage = self.resolve_lineage(lineage_db)

        # Unconditional generation
        if self.rna_type is None and resolved_lineage is None:
            return "<bos>5"

        # Build condition parts
        condition_parts = []

        # Species condition (lineage comes first)
        if resolved_lineage is not None:
            condition_parts.append(resolved_lineage)

        # RNA type condition
        if self.rna_type is not None:
            rna_token = get_rna_token(self.rna_type)
            condition_parts.append(rna_token)

        # Combine conditions
        # Format: <bos>|condition1;condition2|5
        condition_str = ";".join(condition_parts)
        return f"<bos>|{condition_str}|5"

    def build_glm_prompt(
        self,
        prefix: str,
        suffix: str,
        span_id: int,
        lineage_db: Optional[LineageDatabase] = None
    ) -> str:
        """
        Build GLM Span Infilling prompt

        Strict reference to format from line 67 of ORIGINAL_SCRIPT_ANALYSIS.md:
            <bos_glm>|<rna_mRNA>|5[prefix]<span_X>[suffix]3<eos><span_X>

        Args:
            prefix: Sequence before the span
            suffix: Sequence after the span
            span_id: Span marker ID (0-49)
            lineage_db: Lineage database instance

        Returns:
            GLM prompt string
        """
        # Validate conditions
        self.validate()

        # Validate span_id
        if not 0 <= span_id <= 49:
            raise ValueError(f"span_id must be in range 0-49, current value: {span_id}")

        # Resolve lineage
        # If lineage is directly specified, use it; otherwise resolve through lineage_db
        resolved_lineage = None
        if self.lineage is not None:
            resolved_lineage = self.lineage
        elif lineage_db is not None:
            resolved_lineage = self.resolve_lineage(lineage_db)

        # Build condition parts
        condition_parts = []

        if resolved_lineage is not None:
            condition_parts.append(resolved_lineage)

        if self.rna_type is not None:
            rna_token = get_rna_token(self.rna_type)
            condition_parts.append(rna_token)

        # Build prompt
        span_token = f"<span_{span_id}>"

        if condition_parts:
            condition_str = ";".join(condition_parts)
            prompt = f"<bos_glm>|{condition_str}|5{prefix}{span_token}{suffix}3<eos>{span_token}"
        else:
            # Unconditional GLM
            prompt = f"<bos_glm>5{prefix}{span_token}{suffix}3<eos>{span_token}"

        return prompt

    def is_conditional(self) -> bool:
        """Check if this is conditional generation"""
        return (
            self.rna_type is not None or
            self.species is not None or
            self.taxid is not None or
            self.lineage is not None
        )

    def __repr__(self) -> str:
        parts = []
        if self.rna_type:
            parts.append(f"rna_type='{self.rna_type}'")
        if self.taxid:
            parts.append(f"taxid='{self.taxid}'")
        if self.species:
            parts.append(f"species='{self.species}'")
        if self.lineage:
            parts.append(f"lineage='{self.lineage[:30]}...'")
        return f"GenerationCondition({', '.join(parts) if parts else 'unconditional'})"
