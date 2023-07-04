import pywt
import matplotlib.pyplot as plt

[phi, psi, x] = pywt.Wavelet('sym4').wavefun(level=4)

plt.plot(phi)
plt.savefig("symlet_phi.png", dpi=300, bbox_inches="tight")
plt.show()

plt.plot(psi)
plt.savefig("symlet_psi.png", dpi=300, bbox_inches="tight")
plt.show()
