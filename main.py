import mitsuba as mi
import matplotlib.pyplot as plt
mi.set_variant('cuda_ad_rgb')


scene = mi.load_file("assets/cornell-box/scene.xml")
image = mi.render(scene, spp=16)

plt.axis("off")
plt.imshow(image ** (1.0 / 2.2))
plt.show()