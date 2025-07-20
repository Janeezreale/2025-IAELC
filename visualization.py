import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Define data loading function
def process_and_plot(file_path, title_suffix=""):
    df = pd.read_csv(file_path)

    # Select and clean columns
    selected_df = df[['Character', 'Goodness', 'Badness']]
    selected_df = selected_df.dropna(subset=['Character', 'Goodness', 'Badness'])
    print(selected_df.head())

    # Calculate Ambivalence scores by Goodness/Badness scores
    df['Ethicality'] = selected_df['Goodness'] - selected_df['Badness']
    df['Ambivalence'] = selected_df[['Goodness', 'Badness']].min(axis=1) * 2

    # Find optimal k value by silhouette score
    optimal_k = 0
    best_score = -1

    for k in range(3, len(df)+1): # 2 is too small
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(df[['Ethicality', 'Ambivalence']])
        score = silhouette_score(df[['Ethicality', 'Ambivalence']], labels)
        print(f"k = {k}, silhouette score = {score:.3f}")
        if score > best_score:
            optimal_k = k
            best_score = score
#        if optimal_k != 11:
#            optimal_k = 11 # Practically, the optimal val for presentation was 11

    print(f"[{file_path}] Optimal k by silhouette score: {optimal_k}")

    # Perform clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=0)
    df['Cluster'] = kmeans.fit_predict(df[['Ethicality', 'Ambivalence']])

    # Group characters by cluster and exact coordinates
    grouped = df.groupby(['Ethicality', 'Ambivalence', 'Cluster']).agg({
        'Character': lambda x: list(x)
    }).reset_index()

    # Plotting settings
    sns.set(style="whitegrid", font_scale=1.2)
    palette = sns.color_palette("Set3", n_colors=optimal_k)

    plt.figure(figsize=(12, 8))
    legend_handles = []

    # Plot each point
    for _, row in grouped.iterrows():
        ethicality = row['Ethicality']
        ambivalence = row['Ambivalence']
        cluster_id = row['Cluster']
        characters = row['Character']
        
        size = ambivalence * 70 # larger ambivalence → bigger point
        color = palette[cluster_id % len(palette)]
        label = ', '.join(characters)
        
        sc = plt.scatter(ethicality, ambivalence,
            s=size, color=color, alpha=0.9,
            edgecolor='black', linewidth=1.2,
            label=label)
        
        exists = any([h.get_label() == label for h in legend_handles])
        if not exists:
            legend_handles.append(sc)

    plt.xlabel("Ethicality", fontsize=11)
    plt.ylabel("Ambivalence", fontsize=11)
    plt.title(f"Digital Map of Ethical Ambivalence {title_suffix}", fontsize=14)
    plt.grid(alpha=0.3)

    # Create legend
    plt.legend(handles=legend_handles, title="Character",
        fontsize=8, title_fontsize=9,
        loc='upper center', bbox_to_anchor=(0.5, -0.15),
        ncol=3, frameon=True,
        handletextpad=0.3, columnspacing=0.7, borderaxespad=0.2)

    plt.tight_layout()
    plt.show()

# Load files (notice your current file path)
file_list = [("./ambivalence_classic.csv", "(Classical Heroes)"),
            ("./ambivalence_modern.csv", "(Modern Heroes)")]

# Run
for file_path, title_suffix in file_list:
    process_and_plot(file_path, title_suffix)
