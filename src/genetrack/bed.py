import matplotlib.pyplot as plt
import numpy as np

def plot_bed12(chrom, start, end, name, strand, thick_start, thick_end, 
               block_starts, block_sizes, ax, y_pos):
    """
    Plot a single BED12 entry with more strand arrows between exons
    """
    # Plot thin regions (UTRs)
    ax.plot([start, end], [y_pos, y_pos], color='gray', linewidth=1)
    
    # Plot thick regions (CDS)
    if thick_start != thick_end:
        ax.plot([thick_start, thick_end], [y_pos, y_pos], 
                color='blue', linewidth=0.01)
    
    # Plot exons (blocks) and store their positions
    exon_positions = []
    for block_start, block_size in zip(block_starts, block_sizes):
        exon_start = start + block_start
        exon_end = exon_start + block_size
        ax.add_patch(plt.Rectangle((exon_start, y_pos-0.1), 
                                 block_size, 0.2, 
                                 color='blue'))
        exon_positions.append((exon_start, exon_end))
    
    # Add more small arrows between exons
    if len(exon_positions) > 1:  # Only add arrows if there are multiple exons
        for i in range(len(exon_positions)-1):
            gap_start = exon_positions[i][1]  # End of current exon
            gap_end = exon_positions[i+1][0]  # Start of next exon
            gap_size = gap_end - gap_start
            
            if gap_size > 30:  # Lowered threshold for adding arrows
                num_arrows = min(int(gap_size / 50), 10)  # More frequent (50 instead of 100) and max 10 arrows
                if num_arrows > 0:
                    arrow_positions = np.linspace(gap_start + 15, gap_end - 15, num_arrows)
                    for pos in arrow_positions:
                        if strand == '+':
                            ax.arrow(pos, y_pos, 10, 0, head_width=0.05, 
                                   head_length=8, color='black', linewidth=0.5)
                        elif strand == '-':
                            ax.arrow(pos, y_pos, -10, 0, head_width=0.05, 
                                   head_length=8, color='black', linewidth=0.5)
        # Plot thick regions (CDS)
    if thick_start != thick_end:
        ax.plot([thick_start, thick_end], [y_pos, y_pos], 
                color='black', linewidth=1)
    # Add gene name
    ax.text(start, y_pos + 0.2, name, fontsize=8)

# Example usage with sample data
fig, ax = plt.subplots(figsize=(12, 4))

# Sample BED12 data
sample_gene = {
    'chrom': 'chr1',
    'start': 1000,
    'end': 5000,
    'name': 'GeneA',
    'strand': '-',
    'thick_start': 1500,
    'thick_end': 4500,
    'block_starts': [0, 2000, 3500],
    'block_sizes': [500, 1000, 500]
}

plot_bed12(
    sample_gene['chrom'],
    sample_gene['start'],
    sample_gene['end'],
    sample_gene['name'],
    sample_gene['strand'],
    sample_gene['thick_start'],
    sample_gene['thick_end'],
    sample_gene['block_starts'],
    sample_gene['block_sizes'],
    ax,
    y_pos=1
)

# Customize the plot
ax.set_xlim(sample_gene['start']-500, sample_gene['end']+500)
ax.set_ylim(0, 2)
ax.set_xlabel('Genomic Position')
ax.set_title(f'Gene Browser View - {sample_gene["chrom"]}')
ax.set_yticks([])

plt.tight_layout()
plt.show()