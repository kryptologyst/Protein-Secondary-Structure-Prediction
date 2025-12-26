"""Protein sequence and structure utilities."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

# Optional Bio imports
try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    BIO_AVAILABLE = True
except ImportError:
    BIO_AVAILABLE = False


# Standard amino acid alphabet
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AMINO_ACID_TO_IDX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
IDX_TO_AMINO_ACID = {i: aa for i, aa in enumerate(AMINO_ACIDS)}

# Secondary structure classes
SECONDARY_STRUCTURES = ["H", "E", "C"]  # Helix, Sheet, Coil
STRUCTURE_TO_IDX = {s: i for i, s in enumerate(SECONDARY_STRUCTURES)}
IDX_TO_STRUCTURE = {i: s for i, s in enumerate(SECONDARY_STRUCTURES)}

# Amino acid properties
AMINO_ACID_PROPERTIES = {
    "A": {"hydrophobic": True, "polar": False, "charged": False, "aromatic": False},
    "C": {"hydrophobic": True, "polar": False, "charged": False, "aromatic": False},
    "D": {"hydrophobic": False, "polar": True, "charged": True, "aromatic": False},
    "E": {"hydrophobic": False, "polar": True, "charged": True, "aromatic": False},
    "F": {"hydrophobic": True, "polar": False, "charged": False, "aromatic": True},
    "G": {"hydrophobic": True, "polar": False, "charged": False, "aromatic": False},
    "H": {"hydrophobic": False, "polar": True, "charged": True, "aromatic": True},
    "I": {"hydrophobic": True, "polar": False, "charged": False, "aromatic": False},
    "K": {"hydrophobic": False, "polar": True, "charged": True, "aromatic": False},
    "L": {"hydrophobic": True, "polar": False, "charged": False, "aromatic": False},
    "M": {"hydrophobic": True, "polar": False, "charged": False, "aromatic": False},
    "N": {"hydrophobic": False, "polar": True, "charged": False, "aromatic": False},
    "P": {"hydrophobic": True, "polar": False, "charged": False, "aromatic": False},
    "Q": {"hydrophobic": False, "polar": True, "charged": False, "aromatic": False},
    "R": {"hydrophobic": False, "polar": True, "charged": True, "aromatic": False},
    "S": {"hydrophobic": False, "polar": True, "charged": False, "aromatic": False},
    "T": {"hydrophobic": False, "polar": True, "charged": False, "aromatic": False},
    "V": {"hydrophobic": True, "polar": False, "charged": False, "aromatic": False},
    "W": {"hydrophobic": True, "polar": False, "charged": False, "aromatic": True},
    "Y": {"hydrophobic": False, "polar": True, "charged": False, "aromatic": True},
}


def encode_sequence(sequence: str) -> torch.Tensor:
    """Encode protein sequence as tensor of amino acid indices.
    
    Args:
        sequence: Protein sequence string.
        
    Returns:
        Tensor of amino acid indices.
    """
    return torch.tensor(
        [AMINO_ACID_TO_IDX[aa] for aa in sequence if aa in AMINO_ACID_TO_IDX],
        dtype=torch.long,
    )


def decode_sequence(encoded: torch.Tensor) -> str:
    """Decode tensor of amino acid indices back to sequence string.
    
    Args:
        encoded: Tensor of amino acid indices.
        
    Returns:
        Protein sequence string.
    """
    return "".join([IDX_TO_AMINO_ACID[idx.item()] for idx in encoded])


def encode_structure(structure: str) -> torch.Tensor:
    """Encode secondary structure as tensor of structure indices.
    
    Args:
        structure: Secondary structure string.
        
    Returns:
        Tensor of structure indices.
    """
    return torch.tensor(
        [STRUCTURE_TO_IDX[s] for s in structure if s in STRUCTURE_TO_IDX],
        dtype=torch.long,
    )


def decode_structure(encoded: torch.Tensor) -> str:
    """Decode tensor of structure indices back to structure string.
    
    Args:
        encoded: Tensor of structure indices.
        
    Returns:
        Secondary structure string.
    """
    return "".join([IDX_TO_STRUCTURE[idx.item()] for idx in encoded])


def get_amino_acid_features(sequence: str) -> np.ndarray:
    """Get amino acid property features for a sequence.
    
    Args:
        sequence: Protein sequence string.
        
    Returns:
        Feature matrix of shape (seq_len, num_features).
    """
    features = []
    for aa in sequence:
        if aa in AMINO_ACID_PROPERTIES:
            props = AMINO_ACID_PROPERTIES[aa]
            features.append([
                float(props["hydrophobic"]),
                float(props["polar"]),
                float(props["charged"]),
                float(props["aromatic"]),
            ])
        else:
            features.append([0.0, 0.0, 0.0, 0.0])
    
    return np.array(features)


def validate_sequence(sequence: str) -> bool:
    """Validate that sequence contains only standard amino acids.
    
    Args:
        sequence: Protein sequence string.
        
    Returns:
        True if sequence is valid.
    """
    return all(aa in AMINO_ACIDS for aa in sequence)


def validate_structure(structure: str) -> bool:
    """Validate that structure contains only valid secondary structure labels.
    
    Args:
        structure: Secondary structure string.
        
    Returns:
        True if structure is valid.
    """
    return all(s in SECONDARY_STRUCTURES for s in structure)


def pad_sequences(
    sequences: List[torch.Tensor], 
    max_length: Optional[int] = None,
    pad_value: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad sequences to the same length.
    
    Args:
        sequences: List of sequence tensors.
        max_length: Maximum length to pad to. If None, use max length in batch.
        pad_value: Value to use for padding.
        
    Returns:
        Tuple of (padded_sequences, attention_mask).
    """
    if max_length is None:
        max_length = max(seq.size(0) for seq in sequences)
    
    batch_size = len(sequences)
    padded = torch.full((batch_size, max_length), pad_value, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
    
    for i, seq in enumerate(sequences):
        length = min(seq.size(0), max_length)
        padded[i, :length] = seq[:length]
        attention_mask[i, :length] = True
    
    return padded, attention_mask


def load_fasta(file_path: str) -> List:
    """Load protein sequences from FASTA file.
    
    Args:
        file_path: Path to FASTA file.
        
    Returns:
        List of SeqRecord objects.
    """
    if not BIO_AVAILABLE:
        raise ImportError("BioPython is required for FASTA file operations. Install with: pip install biopython")
    return list(SeqIO.parse(file_path, "fasta"))


def save_fasta(sequences: List, file_path: str) -> None:
    """Save protein sequences to FASTA file.
    
    Args:
        sequences: List of SeqRecord objects.
        file_path: Path to save FASTA file.
    """
    if not BIO_AVAILABLE:
        raise ImportError("BioPython is required for FASTA file operations. Install with: pip install biopython")
    SeqIO.write(sequences, file_path, "fasta")


def calculate_sequence_similarity(seq1: str, seq2: str) -> float:
    """Calculate sequence similarity between two protein sequences.
    
    Args:
        seq1: First protein sequence.
        seq2: Second protein sequence.
        
    Returns:
        Similarity score between 0 and 1.
    """
    if len(seq1) != len(seq2):
        return 0.0
    
    matches = sum(1 for a, b in zip(seq1, seq2) if a == b)
    return matches / len(seq1)


def get_sequence_statistics(sequence: str) -> Dict[str, Union[int, float]]:
    """Get basic statistics for a protein sequence.
    
    Args:
        sequence: Protein sequence string.
        
    Returns:
        Dictionary of sequence statistics.
    """
    length = len(sequence)
    aa_counts = {aa: sequence.count(aa) for aa in AMINO_ACIDS}
    aa_frequencies = {aa: count / length for aa, count in aa_counts.items()}
    
    # Calculate amino acid composition properties
    hydrophobic_count = sum(
        count for aa, count in aa_counts.items() 
        if AMINO_ACID_PROPERTIES[aa]["hydrophobic"]
    )
    polar_count = sum(
        count for aa, count in aa_counts.items() 
        if AMINO_ACID_PROPERTIES[aa]["polar"]
    )
    charged_count = sum(
        count for aa, count in aa_counts.items() 
        if AMINO_ACID_PROPERTIES[aa]["charged"]
    )
    aromatic_count = sum(
        count for aa, count in aa_counts.items() 
        if AMINO_ACID_PROPERTIES[aa]["aromatic"]
    )
    
    return {
        "length": length,
        "amino_acid_counts": aa_counts,
        "amino_acid_frequencies": aa_frequencies,
        "hydrophobic_fraction": hydrophobic_count / length,
        "polar_fraction": polar_count / length,
        "charged_fraction": charged_count / length,
        "aromatic_fraction": aromatic_count / length,
    }
