"""Streamlit demo for protein structure prediction."""

import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from src.models.protein_models import create_model
from src.utils.protein import (
    encode_sequence,
    decode_sequence,
    decode_structure,
    validate_sequence,
    get_sequence_statistics,
    AMINO_ACIDS,
    SECONDARY_STRUCTURES,
)
from src.utils.core import get_device


# Page configuration
st.set_page_config(
    page_title="Protein Structure Prediction",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Research Demo Disclaimer</h4>
    <p><strong>This is a research demonstration tool only.</strong></p>
    <ul>
        <li>Not intended for clinical or diagnostic use</li>
        <li>Results should not be used for medical decision-making</li>
        <li>Always consult qualified healthcare professionals for medical advice</li>
        <li>This tool is for educational and research purposes only</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🧬 Protein Secondary Structure Prediction</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Model Configuration")

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    ["bilstm", "transformer", "cnnlstm"],
    help="Choose the neural network architecture"
)

# Model parameters
st.sidebar.subheader("Model Parameters")

if model_type == "bilstm":
    embed_dim = st.sidebar.slider("Embedding Dimension", 64, 256, 128)
    hidden_dim = st.sidebar.slider("Hidden Dimension", 128, 512, 256)
    num_layers = st.sidebar.slider("Number of Layers", 1, 4, 2)
elif model_type == "transformer":
    embed_dim = st.sidebar.slider("Embedding Dimension", 128, 512, 256)
    num_heads = st.sidebar.slider("Number of Attention Heads", 4, 16, 8)
    num_layers = st.sidebar.slider("Number of Layers", 2, 12, 6)
elif model_type == "cnnlstm":
    embed_dim = st.sidebar.slider("Embedding Dimension", 64, 256, 128)
    cnn_channels = st.sidebar.slider("CNN Channels", 32, 128, 64)
    lstm_hidden_dim = st.sidebar.slider("LSTM Hidden Dimension", 128, 512, 256)

dropout = st.sidebar.slider("Dropout Rate", 0.0, 0.5, 0.3)
include_features = st.sidebar.checkbox("Include Amino Acid Features", False)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Protein Sequence")
    
    # Sequence input
    sequence_input = st.text_area(
        "Enter protein sequence (single letter amino acid codes):",
        value="MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        height=100,
        help="Enter a protein sequence using standard single-letter amino acid codes (A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)"
    )
    
    # Validate sequence
    if sequence_input:
        sequence = sequence_input.strip().upper()
        if validate_sequence(sequence):
            st.success(f"✅ Valid sequence (length: {len(sequence)})")
            
            # Sequence statistics
            stats = get_sequence_statistics(sequence)
            
            st.subheader("Sequence Statistics")
            col_stats1, col_stats2 = st.columns(2)
            
            with col_stats1:
                st.metric("Length", stats["length"])
                st.metric("Hydrophobic Fraction", f"{stats['hydrophobic_fraction']:.3f}")
            
            with col_stats2:
                st.metric("Polar Fraction", f"{stats['polar_fraction']:.3f}")
                st.metric("Charged Fraction", f"{stats['charged_fraction']:.3f}")
            
            # Amino acid composition
            st.subheader("Amino Acid Composition")
            aa_freq = stats["amino_acid_frequencies"]
            aa_df = pd.DataFrame(list(aa_freq.items()), columns=["Amino Acid", "Frequency"])
            aa_df = aa_df.sort_values("Frequency", ascending=False)
            
            fig_aa = go.Figure(data=[
                go.Bar(x=aa_df["Amino Acid"], y=aa_df["Frequency"])
            ])
            fig_aa.update_layout(
                title="Amino Acid Frequency Distribution",
                xaxis_title="Amino Acid",
                yaxis_title="Frequency",
                height=300
            )
            st.plotly_chart(fig_aa, use_container_width=True)
            
        else:
            st.error("❌ Invalid sequence. Please use only standard amino acid codes.")
            sequence = None
    else:
        sequence = None

with col2:
    st.subheader("Prediction Results")
    
    if sequence and len(sequence) > 0:
        # Create model
        try:
            model_params = {
                "vocab_size": 20,
                "num_classes": 3,
                "max_length": 512,
                "include_features": include_features,
                "dropout": dropout,
            }
            
            if model_type == "bilstm":
                model_params.update({
                    "embed_dim": embed_dim,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                })
            elif model_type == "transformer":
                model_params.update({
                    "embed_dim": embed_dim,
                    "num_heads": num_heads,
                    "num_layers": num_layers,
                    "hidden_dim": embed_dim * 2,
                })
            elif model_type == "cnnlstm":
                model_params.update({
                    "embed_dim": embed_dim,
                    "cnn_channels": cnn_channels,
                    "lstm_hidden_dim": lstm_hidden_dim,
                    "num_lstm_layers": 2,
                })
            
            model = create_model(model_type, **model_params)
            model.eval()
            
            # Prepare input
            encoded_seq = encode_sequence(sequence)
            seq_tensor = torch.zeros(512, dtype=torch.long)
            seq_tensor[:len(encoded_seq)] = encoded_seq
            
            attention_mask = torch.zeros(512, dtype=torch.bool)
            attention_mask[:len(encoded_seq)] = True
            
            batch = {
                "sequence": seq_tensor.unsqueeze(0),
                "attention_mask": attention_mask.unsqueeze(0),
            }
            
            if include_features:
                from src.utils.protein import get_amino_acid_features
                features = get_amino_acid_features(sequence)
                features_tensor = torch.zeros(512, features.shape[1])
                features_tensor[:len(features)] = torch.tensor(features, dtype=torch.float)
                batch["features"] = features_tensor.unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                logits = model(batch)
                predictions = logits.argmax(dim=-1)
                probabilities = torch.softmax(logits, dim=-1)
            
            # Decode predictions
            pred_structure = decode_structure(predictions[0, :len(sequence)])
            
            # Display results
            st.success("🎯 Prediction completed!")
            
            # Structure visualization
            st.subheader("Predicted Secondary Structure")
            
            # Create structure visualization
            structure_colors = {"H": "#ff6b6b", "E": "#4ecdc4", "C": "#45b7d1"}
            structure_names = {"H": "Helix", "E": "Sheet", "C": "Coil"}
            
            # Create sequence and structure display
            seq_display = ""
            struct_display = ""
            color_display = ""
            
            for i, (aa, struct) in enumerate(zip(sequence, pred_structure)):
                seq_display += aa
                struct_display += struct
                color_display += f"background-color: {structure_colors[struct]}; color: white; padding: 2px; margin: 1px;"
            
            # Display sequence
            st.markdown("**Sequence:**")
            st.code(sequence)
            
            # Display structure
            st.markdown("**Predicted Structure:**")
            structure_text = ""
            for struct in pred_structure:
                structure_text += structure_names[struct][0]  # First letter
            
            st.code(structure_text)
            
            # Structure statistics
            st.subheader("Structure Statistics")
            struct_counts = {name: pred_structure.count(symbol) for symbol, name in structure_names.items()}
            struct_fractions = {name: count / len(pred_structure) for name, count in struct_counts.items()}
            
            col_struct1, col_struct2, col_struct3 = st.columns(3)
            
            with col_struct1:
                st.metric("Helix", f"{struct_counts['Helix']} ({struct_fractions['Helix']:.1%})")
            with col_struct2:
                st.metric("Sheet", f"{struct_counts['Sheet']} ({struct_fractions['Sheet']:.1%})")
            with col_struct3:
                st.metric("Coil", f"{struct_counts['Coil']} ({struct_fractions['Coil']:.1%})")
            
            # Confidence visualization
            st.subheader("Prediction Confidence")
            
            # Get probabilities for valid positions
            valid_probs = probabilities[0, :len(sequence), :].numpy()
            
            # Create confidence plot
            fig_conf = make_subplots(
                rows=3, cols=1,
                subplot_titles=["Helix Confidence", "Sheet Confidence", "Coil Confidence"],
                vertical_spacing=0.1
            )
            
            positions = list(range(len(sequence)))
            
            for i, (struct_name, color) in enumerate([("Helix", "#ff6b6b"), ("Sheet", "#4ecdc4"), ("Coil", "#45b7d1")]):
                fig_conf.add_trace(
                    go.Scatter(
                        x=positions,
                        y=valid_probs[:, i],
                        mode='lines+markers',
                        name=struct_name,
                        line=dict(color=color, width=2),
                        marker=dict(size=4)
                    ),
                    row=i+1, col=1
                )
            
            fig_conf.update_layout(
                height=600,
                showlegend=False,
                title="Prediction Confidence by Position"
            )
            
            fig_conf.update_xaxes(title_text="Position", row=3, col=1)
            fig_conf.update_yaxes(title_text="Probability", row=2, col=1)
            
            st.plotly_chart(fig_conf, use_container_width=True)
            
            # Model information
            st.subheader("Model Information")
            st.info(f"""
            **Model Type:** {model_type.upper()}
            **Sequence Length:** {len(sequence)}
            **Parameters:** {sum(p.numel() for p in model.parameters()):,}
            **Features Used:** {'Yes' if include_features else 'No'}
            """)
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {str(e)}")
            st.info("This might be due to model configuration issues. Try adjusting the parameters.")
    
    else:
        st.info("👈 Please enter a valid protein sequence to see predictions")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🧬 Protein Structure Prediction Demo | Research Use Only | Not for Clinical Use</p>
    <p>Built with PyTorch, Streamlit, and Plotly</p>
</div>
""", unsafe_allow_html=True)
