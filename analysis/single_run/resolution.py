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
bal_acc_large = [0.947693722090885,
    0.932320433120275,
    0.940311268715524,
    0.920317543706038,
    0.920967224119312
]

bal_acc_small = [
    0.947268190175991,
    0.935631001371742,
    0.946948311589761,
    0.938626798587397,
    0.918454747103289
]

macroF1_large = [
    0.947765317506417,
    0.929316095445766,
    0.934556952649518,
    0.916888702912848,
    0.920931531787215
]

macroF1_small = [
    0.947332699205358,
    0.933940107055939,
    0.939895991418745,
    0.932140174846569,
    0.912182354836798
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