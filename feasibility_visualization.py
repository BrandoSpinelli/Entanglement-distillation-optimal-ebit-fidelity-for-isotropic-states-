import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
import matplotlib.patches as mpatches

def create_ieee_feasibility_plot():
    """
    Crea un grafico a barre elegante in stile IEEE che mostra i risultati
    della feasibility del vettore b per diversi valori di k.
    """
    
    # Dati dalla tabella riassuntiva
    data = {
        'k=2': [Fraction(1,2), Fraction(-1,3), Fraction(5,24)],
        'k=3': [Fraction(1,2), Fraction(-1,3), Fraction(1,6), Fraction(0,1)],
        'k=4': [Fraction(1,2), Fraction(-17,48), Fraction(7,36), Fraction(-13,192), Fraction(0,1)],
        'k=5': [Fraction(1,2), Fraction(-17,72), Fraction(17,144), Fraction(-1,36), Fraction(7,576), Fraction(0,1)],
        'k=6': [Fraction(1,2), Fraction(-49,144), Fraction(49,576), Fraction(-25,1728), Fraction(41,3456), Fraction(-61,10368), Fraction(0,1)]
    }
    
    # Converti in float per il plotting
    data_float = {}
    for k, values in data.items():
        data_float[k] = [float(frac) for frac in values]
    
    # Configurazione figura
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Colori IEEE style
    colors = ['#1f4e79', '#2e8b57', '#8b4513', '#4682b4', '#32cd32']
    
    # ========== GRAFICO 1: Valori S_j(b) per ogni k ==========
    
    # Preparazione dati per il plot raggruppato
    max_j = max(len(values) for values in data_float.values())
    x_positions = np.arange(max_j)
    width = 0.15
    
    for i, (k_label, values) in enumerate(data_float.items()):
        # Estendi i valori con NaN per uniformare la lunghezza
        extended_values = values + [np.nan] * (max_j - len(values))
        
        # Posizioni delle barre per questo k
        positions = x_positions + i * width - 2 * width
        
        # Crea le barre
        bars = ax1.bar(positions, extended_values, width, 
                      label=k_label, color=colors[i], alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        
        # Aggiungi valori numerici sopra le barre
        for j, (pos, val) in enumerate(zip(positions, extended_values)):
            if not np.isnan(val) and abs(val) > 1e-10:  # Solo se non è NaN o zero
                # Formatta il valore come frazione
                frac = data[k_label][j]
                if frac.denominator == 1:
                    text = str(frac.numerator)
                else:
                    text = f"{frac.numerator}/{frac.denominator}"
                
                # Posiziona il testo
                y_offset = 0.01 if val >= 0 else -0.03
                ax1.text(pos, val + y_offset, text, ha='center', va='bottom' if val >= 0 else 'top',
                        fontsize=7, fontweight='bold')
    
    # Linee di riferimento per i limiti di feasibility
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Limite superiore')
    ax1.axhline(y=-0.5, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='Limite inferiore')
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=0.8)
    
    ax1.set_xlabel('Indice j', fontsize=12, fontweight='bold')
    ax1.set_ylabel('$S_j(b)$', fontsize=12, fontweight='bold')
    ax1.set_title('Valori di Feasibility $S_j(b)$ per il Vettore Ottimale', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([f'j={i}' for i in range(max_j)])
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(-0.6, 0.6)
    
    # ========== GRAFICO 2: Analisi del margine di feasibility ==========
    
    # Calcola quanto ogni valore è vicino ai limiti
    margins = {}
    for k_label, values in data_float.items():
        margin_list = []
        for val in values:
            if not np.isnan(val):
                # Distanza dal limite più vicino
                dist_upper = 0.5 - abs(val)
                dist_lower = abs(val) - (-0.5)
                margin = min(dist_upper, dist_lower)
                margin_list.append(margin)
            else:
                margin_list.append(np.nan)
        margins[k_label] = margin_list
    
    # Plot dei margini
    for i, (k_label, margin_values) in enumerate(margins.items()):
        positions = x_positions[:len(margin_values)] + i * width - 2 * width
        
        bars = ax2.bar(positions, margin_values, width,
                      label=k_label, color=colors[i], alpha=0.8,
                      edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Indice j', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Margine di Feasibility', fontsize=12, fontweight='bold')
    ax2.set_title('Margine di Sicurezza rispetto ai Limiti di Feasibility', 
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x_positions)
    ax2.set_xticklabels([f'j={i}' for i in range(max_j)])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 0.6)
    
    # Aggiunge una zona di "sicurezza"
    ax2.axhspan(0, 0.1, alpha=0.2, color='red', label='Zona critica')
    ax2.axhspan(0.1, 0.6, alpha=0.1, color='green', label='Zona sicura')
    
    plt.tight_layout()
    plt.show()

def create_summary_heatmap():
    """
    Crea una heatmap che mostra tutti i valori S_j(b) in forma compatta
    """
    
    # Dati in formato float
    data_matrix = []
    k_labels = []
    
    # Dati dalla tabella
    results = {
        2: [0.5, -1/3, 5/24],
        3: [0.5, -1/3, 1/6, 0],
        4: [0.5, -17/48, 7/36, -13/192, 0],
        5: [0.5, -17/72, 17/144, -1/36, 7/576, 0],
        6: [0.5, -49/144, 49/576, -25/1728, 41/3456, -61/10368, 0]
    }
    
    # Trova la dimensione massima
    max_len = max(len(values) for values in results.values())
    
    # Crea la matrice con padding di NaN
    for k in sorted(results.keys()):
        values = results[k]
        padded = values + [np.nan] * (max_len - len(values))
        data_matrix.append(padded)
        k_labels.append(f'k={k}')
    
    data_matrix = np.array(data_matrix)
    
    # Crea la heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Usa una colormap divergente centrata su zero
    im = ax.imshow(data_matrix, cmap='RdBu_r', aspect='auto', 
                   vmin=-0.5, vmax=0.5, interpolation='nearest')
    
    # Aggiungi i valori numerici alle celle
    for i in range(len(k_labels)):
        for j in range(max_len):
            if not np.isnan(data_matrix[i, j]):
                # Formatta come frazione
                val = data_matrix[i, j]
                if abs(val) < 1e-10:
                    text = "0"
                else:
                    # Converte in frazione per display
                    frac = Fraction(val).limit_denominator(10000)
                    if frac.denominator == 1:
                        text = str(frac.numerator)
                    else:
                        text = f"{frac.numerator}/{frac.denominator}"
                
                # Colore del testo basato sul background
                color = 'white' if abs(val) > 0.25 else 'black'
                ax.text(j, i, text, ha='center', va='center', 
                       color=color, fontsize=9, fontweight='bold')
    
    # Configurazione assi
    ax.set_xticks(range(max_len))
    ax.set_xticklabels([f'j={i}' for i in range(max_len)])
    ax.set_yticks(range(len(k_labels)))
    ax.set_yticklabels(k_labels)
    
    ax.set_xlabel('Indice j', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valore di k', fontsize=12, fontweight='bold')
    ax.set_title('Heatmap dei Valori di Feasibility $S_j(b)$', 
                 fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('$S_j(b)$', fontsize=12, fontweight='bold')
    
    # Aggiungi linee di griglia
    ax.set_xticks(np.arange(-0.5, max_len, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(k_labels), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def create_feasibility_validation_plot():
    """
    Crea un grafico che mostra la validazione della condizione di feasibility
    """
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Verifica che tutti i valori siano nel range [-1/2, 1/2]
    k_values = [2, 3, 4, 5, 6]
    max_violations = []
    min_violations = []
    
    results = {
        2: [0.5, -1/3, 5/24],
        3: [0.5, -1/3, 1/6, 0],
        4: [0.5, -17/48, 7/36, -13/192, 0],
        5: [0.5, -17/72, 17/144, -1/36, 7/576, 0],
        6: [0.5, -49/144, 49/576, -25/1728, 41/3456, -61/10368, 0]
    }
    
    for k in k_values:
        values = results[k]
        max_val = max(values)
        min_val = min(values)
        
        max_violations.append(max(0, max_val - 0.5))  # Violazione limite superiore
        min_violations.append(max(0, -0.5 - min_val))  # Violazione limite inferiore
    
    # Plot delle violazioni
    width = 0.35
    x_pos = np.arange(len(k_values))
    
    bars1 = ax.bar(x_pos - width/2, max_violations, width, 
                   label='Violazione limite superiore', color='red', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, min_violations, width,
                   label='Violazione limite inferiore', color='blue', alpha=0.7)
    
    # Linea di riferimento per feasibility perfetta
    ax.axhline(y=0, color='green', linestyle='-', linewidth=2, 
               label='Feasibility perfetta')
    
    ax.set_xlabel('Valore di k', fontsize=12, fontweight='bold')
    ax.set_ylabel('Violazione dei limiti', fontsize=12, fontweight='bold')
    ax.set_title('Validazione della Condizione di Feasibility', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Aggiungi testo di conferma
    ax.text(0.02, 0.98, 'FEASIBILITY VERIFICATA\nTutte le violazioni = 0', 
            transform=ax.transAxes, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

# Esecuzione
if __name__ == "__main__":
    print("Creando grafico a barre dei risultati di feasibility...")
    create_ieee_feasibility_plot()
    
    print("Creando heatmap riassuntiva...")
    create_summary_heatmap()
    
    print("Creando grafico di validazione...")
    create_feasibility_validation_plot()
