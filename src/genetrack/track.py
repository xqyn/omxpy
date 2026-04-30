import pyBigWig
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



# Load GTF file (replace with your GTF file path)
gtf_file = "genes.gtf"  # Update this path
gtf = pd.read_csv(gtf_file, sep="\t", comment="#", header=None,
                  names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attributes"])



# Open the BigWig file
bw = pyBigWig.open("/exports/ana-scarlab/xqnguyen/projects/arend_ATACseq/track_bw/bigWig_merge_RPKM/11_994_D5_B_BSLE.RPKM.bigWig")

# Specify the chromosome and region of interest
chrom = 'X'
start = 73851081 - 30000 # Start position
end = 73851581 + 3000  # End position

# Fetch coverage data from the BigWig file
coverage = bw.values(chrom, start, end)

# Generate x-axis (genomic positions)
positions = np.arange(start, end)


# Check if data was retrieved successfully
if coverage is None:
    print(f"No data found for {chrom}:{start}-{end}")
else:
    # Plot the coverage
    plt.figure(figsize=(10, 4))
    plt.hist(positions, coverage, color='blue', linewidth=0.3)
    plt.fill_between(positions, coverage, color='blue', alpha=0.3)
    plt.title(f"Coverage Map: {chrom}:{start}-{end}")
    plt.xlabel("Genomic Position")
    plt.ylabel("Coverage")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('testingX.png')
    plt.show()

# Close the BigWig file
bw.close()


# --------------------------------------------------
# Load GTF file (replace with your GTF file path)
#/exports/ana-scarlab/group_references/ensembl/human/102
gtf_file="/exports/ana-scarlab/xqnguyen/projects/arend_ATACseq/track_bw/bigWig_merge_RPKM/gencode_gtf_v38.bed"

gtf = pd.read_csv(gtf_file, sep="\t", comment="#", header=None,
                  names=["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attributes"])
gtf = pd.read_csv(gtf_file, sep="\t", comment="#", header=None)

# generate BED12 format
gtf.columns = ['chr', 'start', 'end', 'gene', 'score', 'strand',
               'thickStart', 'thickEnd', 'itemRgb', 'blockCount', 'blockSizes', 'blockStarts']

# Filter GTF for genes in the specified region
genes = gtf[(gtf["chr"] == chrom) & 
            (gtf["start"] <= end) & 
            (gtf["end"] >= start)]


# --------------------------------------------------
# Create a figure with two subplots: coverage (top) and genes (bottom)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), 
                               height_ratios=[3, 1], sharex=True)

# Plot coverage on the top axis
if coverage is None:
    print(f"No data found for {chrom}:{start}-{end}")
else:
    ax1.plot(positions, coverage, color='blue', linewidth=1)
    ax1.fill_between(positions, coverage, color='blue', alpha=0.3)
    ax1.set_title(f"Coverage Map: {chrom}:{start}-{end}")
    ax1.set_ylabel("Coverage")
    ax1.grid(True, linestyle='--', alpha=0.7)

# Plot genes on the bottom axis
if genes.empty:
    ax2.text(0.5, 0.5, "No genes found in this region", 
             ha="center", va="center", transform=ax2.transAxes)
else:
    y_pos = 0.5  # Fixed y-position for gene lines
    for _, gene in genes.iterrows():
        gene_start = max(gene["start"], start)
        gene_end = min(gene["end"], end)
        gene_name = gene
        
        # Draw a horizontal line for the gene
        ax2.plot([gene_start, gene_end], [y_pos, y_pos], color="black", linewidth=2)
        # Add gene name above the line
        ax2.text((gene_start + gene_end) / 2, y_pos + 0.1, gene_name, 
                 ha="center", va="bottom", fontsize=8)

# Customize the gene axis
ax2.set_ylim(0, 1)
ax2.set_yticks([])  # Hide y-axis ticks for genes
ax2.set_xlabel("Genomic Position")

# Adjust layout and display
plt.tight_layout()
plt.savefig('testing2.png')
plt.show()

# Close the BigWig file
bw.close()


# --------------------------------------------------
# Load BED12 file
bed = pybedtools.BedTool(gtf_file)

# Plotting logic
fig, ax = plt.subplots(figsize=(10, 2))
for feature in bed:
    start = feature.start
    end = feature.end
    strand = feature.strand
    thick_start = feature.thick_start
    thick_end = feature.thick_end
    blocks = feature.block_sizes
    block_starts = feature.block_starts

    # Plot introns (thin lines)
    ax.plot([start, end], [1, 1], color="gray", linewidth=0.5)
    
    # Plot exons (thick blocks)
    for b_start, b_size in zip(block_starts, blocks):
        ax.fill_between([start + b_start, start + b_start + b_size], 0.8, 1.2, color="blue")

    # Plot coding region (thicker)
    if thick_start != thick_end:
        ax.fill_between([thick_start, thick_end], 0.9, 1.1, color="red")

ax.set_ylim(0, 2)
ax.set_xlim(900, 4600)
ax.set_xlabel("Genomic Position (chr1)")
ax.set_title("Gene Annotations")
plt.savefig("output.png")
plt.show()