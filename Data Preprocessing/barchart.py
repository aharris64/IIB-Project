import matplotlib.pyplot as plt

labels = ['Normal', 'Pseudo-papilloedema', 'Papilloedema']
values = [2510, 364, 660]

rgb_255 = (40, 130, 138)
color_rgb = tuple(c/255 for c in rgb_255)

plt.bar(labels, values, color=color_rgb)

plt.xlabel('Class', fontsize=14)
plt.ylabel('Number of Images', fontsize=14)
plt.title('Dataset Distribution Across Classifications', fontsize=18)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.show()