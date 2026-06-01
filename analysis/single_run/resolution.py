import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

resolutions = [14,28,56,112,224]
bal_acc_large = [0.9476937220908850,
0.9558147272568070,
0.9395727169249620,
0.9314517117590400,
0.9333315821731890

]

bal_acc_small = [0.9472681901759910,
0.9274459329305660,
0.9675159793363100,
0.9475159793363100,
0.8904229051746780

]

macroF1_large = [0.9477653175064170,
0.9561201314167870,
0.9343887622525390,
0.9260307631824660,
0.9386399108138230

]

macroF1_small = [0.9473326992053580,
0.9297551789077210,
0.9593384884815150,
0.9395914882185890,
0.8905108339632140

]

bal_acc_large = bal_acc_large[::-1]
bal_acc_small = bal_acc_small[::-1]
macroF1_large = macroF1_large[::-1]
macroF1_small = macroF1_small[::-1]

x = np.arange(5)

plt.plot(x, bal_acc_large, label="MobileNetV3 Large", marker='o', markersize=6)
plt.plot(x, bal_acc_small, label="MobileNetV3 Small", marker='o', markersize=6)
plt.xticks(x, ['14', '28', '56', '112', '224'])
plt.xlabel("Input Resolution (px)")
plt.ylabel("Balanced Accuracy")
plt.ylim(0.8, 1)
# plt.xscale('log', base=2)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()

plt.plot(x, macroF1_large, label="MobileNetV3 Large", marker='o', markersize=6)
plt.plot(x, macroF1_small, label="MobileNetV3 Small", marker='o', markersize=6)
plt.xticks(x, ['14', '28', '56', '112', '224'])
plt.xlabel("Input Resolution (px)")
plt.ylabel("Macro F1")
plt.ylim(0.8, 1)
# plt.xscale('log', base=2)
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()