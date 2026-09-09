"""
Species Lineage Database

Manages mapping from TaxID/species name to Greengenes format lineage.
Built-in common model organisms, supports extension from external files.

Strict reference: DEFAULT_NCBI_LINEAGE from lines 474-483 of ORIGINAL_SCRIPT_ANALYSIS.md
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Built-in species mapping table
# Reference: lines 474-483 of ORIGINAL_SCRIPT_ANALYSIS.md
# Format: taxid -> (lineage, species_name)
# Lineage format: lowercase, semicolon-separated, e.g., d__eukaryota;p__chordata;...;s__homo_sapiens
DEFAULT_NCBI_LINEAGE: Dict[str, Tuple[str, str]] = {
    '9606': (
        'd__eukaryota;p__chordata;c__mammalia;o__primates;f__hominidae;g__homo;s__homo_sapiens',
        'homo_sapiens'
    ),
    '10090': (
        'd__eukaryota;p__chordata;c__mammalia;o__rodentia;f__muridae;g__mus;s__mus_musculus',
        'mus_musculus'
    ),
    '10116': (
        'd__eukaryota;p__chordata;c__mammalia;o__rodentia;f__muridae;g__rattus;s__rattus_norvegicus',
        'rattus_norvegicus'
    ),
    '7227': (
        'd__eukaryota;p__arthropoda;c__insecta;o__diptera;f__drosophilidae;g__drosophila;s__drosophila_melanogaster',
        'drosophila_melanogaster'
    ),
    '6239': (
        'd__eukaryota;p__nematoda;c__chromadorea;o__rhabditida;f__rhabditidae;g__caenorhabditis;s__caenorhabditis_elegans',
        'caenorhabditis_elegans'
    ),
    '3702': (
        'd__eukaryota;p__streptophyta;c__magnoliopsida;o__brassicales;f__brassicaceae;g__arabidopsis;s__arabidopsis_thaliana',
        'arabidopsis_thaliana'
    ),
    '4932': (
        'd__eukaryota;p__ascomycota;c__saccharomycetes;o__saccharomycetales;f__saccharomycetaceae;g__saccharomyces;s__saccharomyces_cerevisiae',
        'saccharomyces_cerevisiae'
    ),
    '562': (
        'd__bacteria;p__proteobacteria;c__gammaproteobacteria;o__enterobacterales;f__enterobacteriaceae;g__escherichia;s__escherichia_coli',
        'escherichia_coli'
    ),
}


# Reverse mapping from species name to TaxID (automatically generated)
_SPECIES_TO_TAXID: Dict[str, str] = {
    species: taxid for taxid, (_, species) in DEFAULT_NCBI_LINEAGE.items()
}


class LineageDatabase:
    """
    Species Lineage Database

    Manages mapping from TaxID/species name to Greengenes format lineage.
    Built-in common model organisms, supports extension from external files.

    Built-in species include:
        - Homo sapiens (9606)
        - Mus musculus (10090)
        - Rattus norvegicus (10116)
        - Drosophila melanogaster (7227)
        - Caenorhabditis elegans (6239)
        - Arabidopsis thaliana (3702)
        - Saccharomyces cerevisiae (4932)
        - Escherichia coli (562)

    Example:
        # Use built-in data
        db = LineageDatabase()
        lineage = db.get_lineage(taxid="9606")

        # Load extended data
        db = LineageDatabase(extra_file="my_species.tsv")
    """

    def __init__(self, extra_file: Optional[str] = None):
        """
        Initialize lineage database

        Args:
            extra_file: Path to additional lineage mapping file (TSV format)
                Format: taxid<TAB>lineage_greengenes
        """
        # Copy built-in data
        self._taxid_to_lineage: Dict[str, str] = {
            taxid: lineage for taxid, (lineage, _) in DEFAULT_NCBI_LINEAGE.items()
        }
        self._taxid_to_species: Dict[str, str] = {
            taxid: species for taxid, (_, species) in DEFAULT_NCBI_LINEAGE.items()
        }
        self._species_to_taxid: Dict[str, str] = _SPECIES_TO_TAXID.copy()

        # Load additional data
        if extra_file is not None:
            self._load_extra_file(extra_file)

    def _load_extra_file(self, filepath: str) -> None:
        """
        Load additional lineage mapping file

        Args:
            filepath: TSV file path, format: taxid<TAB>lineage_greengenes
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Lineage file does not exist: {filepath}")

        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split('\t')
                if len(parts) < 2:
                    continue

                taxid = parts[0].strip()
                lineage = parts[1].strip()

                # Extract species name from lineage (part after s__)
                species = self._extract_species_from_lineage(lineage)

                self._taxid_to_lineage[taxid] = lineage
                if species:
                    self._taxid_to_species[taxid] = species
                    self._species_to_taxid[species] = taxid

    def _extract_species_from_lineage(self, lineage: str) -> Optional[str]:
        """Extract species name from lineage string"""
        for part in lineage.split(';'):
            if part.startswith('s__'):
                return part[3:]  # Remove 's__' prefix
        return None

    def get_lineage(
        self,
        taxid: Optional[str] = None,
        species: Optional[str] = None
    ) -> Optional[str]:
        """
        Get lineage string for a species

        Args:
            taxid: NCBI Taxonomy ID
            species: Species name (e.g., "homo_sapiens")

        Returns:
            Greengenes format lineage string, or None if not found

        Note:
            taxid takes priority over species
        """
        # taxid takes priority
        if taxid is not None:
            taxid = str(taxid)  # Ensure it's a string
            if taxid in self._taxid_to_lineage:
                return self._taxid_to_lineage[taxid]

        # Try to find by species name
        if species is not None:
            # Normalize species name: lowercase, replace spaces with underscores
            species_normalized = species.lower().replace(' ', '_')
            if species_normalized in self._species_to_taxid:
                taxid = self._species_to_taxid[species_normalized]
                return self._taxid_to_lineage.get(taxid)

        return None

    def get_species_name(self, taxid: str) -> Optional[str]:
        """
        Get species name for a TaxID

        Args:
            taxid: NCBI Taxonomy ID

        Returns:
            Species name, or None if not found
        """
        return self._taxid_to_species.get(str(taxid))

    def get_taxid(self, species: str) -> Optional[str]:
        """
        Get TaxID for a species name

        Args:
            species: Species name

        Returns:
            TaxID, or None if not found
        """
        species_normalized = species.lower().replace(' ', '_')
        return self._species_to_taxid.get(species_normalized)

    def list_species(self) -> List[Dict[str, str]]:
        """
        List all available species

        Returns:
            List of species information, each element contains taxid, species_name, lineage
        """
        result = []
        for taxid, lineage in self._taxid_to_lineage.items():
            species = self._taxid_to_species.get(taxid, '')
            result.append({
                'taxid': taxid,
                'species_name': species,
                'lineage': lineage
            })
        return result

    def __len__(self) -> int:
        """Return the number of species in the database"""
        return len(self._taxid_to_lineage)

    def __contains__(self, key: str) -> bool:
        """Check if taxid or species name is in the database"""
        if key in self._taxid_to_lineage:
            return True
        key_normalized = key.lower().replace(' ', '_')
        return key_normalized in self._species_to_taxid
